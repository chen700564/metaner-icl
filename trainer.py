from utils import tokenid2result
from transformers import Trainer,Seq2SeqTrainer,AutoModel
import torch.nn as nn
import torch
import os
import numpy as np
from utils import getresult
import copy
import tqdm
from torch.optim import AdamW
import gc
import logging
import sys
from transformers.deepspeed import is_deepspeed_zero3_enabled



class MetaTrainer(Seq2SeqTrainer):

    def set_gen_kwargs(self,**gen_kwargs):
        gen_kwargs = gen_kwargs.copy()
        self._gen_kwargs = gen_kwargs
    
    def predict(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = 'test'):
        gen_kwargs = self._gen_kwargs.copy()
        return super().predict(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,**gen_kwargs)

    def compute_loss(self, model, batch, return_outputs=False):
        if 'decoder_input_ids' in batch:
            seq2seq = True
        else:
            seq2seq = False
        
        flag = True
        if self.state.global_step < self.args.warmup_steps or self.state.global_step % 10 != 0 :
            '''
            make training faster
            '''
            flag = False

        if flag and len(batch['iclindex']) > 0:
            fullftqueryfeatures = []
            i = 0
            batch1 = batch['icl'][i]
            batch2 = batch['query'][i]

            ftmodel = copy.deepcopy(model)
            
            if seq2seq:
                for k,v in ftmodel.named_parameters():
                    v.requires_grad=False
                    if 'encoder' in k:
                        v.requires_grad=True
            parameters_to_optimize = list(ftmodel.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                    {'params': [p for n, p in parameters_to_optimize
                                if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
                    {'params': [p for n, p in parameters_to_optimize
                                if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
                ]
            ftoptimizer = AdamW(parameters_to_optimize, lr=1e-4)
            features = self.getupdatefeature(batch1,batch2,ftmodel,ftoptimizer, seq2seq)
            ftqueryfeatures = features[0]
            fullftqueryfeatures.append(ftqueryfeatures)
            fullftqueryfeatures = torch.cat(fullftqueryfeatures,dim=0)    
        else:
            fullftqueryfeatures = None

        if seq2seq:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'decoder_input_ids': batch['decoder_input_ids'],
                'decoder_attention_mask': batch['decoder_attention_mask'],
                'labels': batch['labels']
            }
        else:
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels']
            }
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        loss1 = outputs.loss

        if fullftqueryfeatures is not None:
            if seq2seq:
                features = outputs.encoder_last_hidden_state[batch['iclindex']]
            else:
                features = outputs.last_hidden_state[batch['iclindex']]
            indexs = batch['indexs']
            queryfeatures = []
            i = 0
            index = indexs[i]
            feature = features[i,index]
            queryfeatures.append(feature)
            queryfeatures = torch.cat(queryfeatures,dim=0)

            loss2 = torch.mean(torch.sqrt(torch.sum((fullftqueryfeatures - queryfeatures)**2,dim=-1) ))
        else:
            loss2 = 0
        loss = loss1 + 0.05 * loss2
        return (loss, outputs) if return_outputs else loss

    def getupdatefeature(self, batch,inputs,model,optimizer, seq2seq=True):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            if seq2seq:
                outputs = model.base_model.encoder(**inputs)
            else:
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return last_hidden_states.detach()
    
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self.args.predict_with_generate = True

        self.set_gen_kwargs(max_length=512,num_beams=1,decoder_start_token_id=self.args.prefixid,eos_token_id=self.args.endid)
        generation = self.predict(eval_dataset)
        predictions = generation.predictions
        predictions = predictions[:,1:]
        preds, golds, _, _ = tokenid2result(predictions,self.args.endid,eval_dataset,self.tokenizer)
        prednum = 0
        goldnum = 0
        tt = 0

        for i in range(len(preds)):
            prednum += len(preds[i])
            goldnum += len(golds[i])
            for entity in preds[i]:
                if entity in golds[i]:
                    tt += 1
        p = tt/prednum if prednum > 0 else 0
        r = tt/goldnum if goldnum > 0 else 0
        f1 = 2*p*r/(p+r) if p + r > 0 else 0
        return {metric_key_prefix+'_p': p, metric_key_prefix+'_r': r, metric_key_prefix+'_f1': f1, metric_key_prefix+'_loss': None}
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)


class GenerativeTrainer(Seq2SeqTrainer):
    def set_gen_kwargs(self,**gen_kwargs):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs
    
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = 'eval'):
        gen_kwargs = self._gen_kwargs.copy()
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,**gen_kwargs)
    
    def predict(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = 'test'):
        gen_kwargs = self._gen_kwargs.copy()
        return super().predict(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,**gen_kwargs)


class GeneratedF1():
    def __init__(self,tokenizer,targets,endid,dataset) -> None:
        self.tokenizer = tokenizer
        self.targets = targets
        self.endid = endid
        self.dataset = dataset


    def __call__(self, eval_preds):
        predictions, _ = eval_preds

        preds, golds, _, _ = tokenid2result(predictions,self.endid,self.dataset,self.tokenizer)

        tt = 0
        prednum = 0
        goldnum = 0

        for instance in preds:
            prednum += 1
            if instance in golds:
                tt += 1
        for instance in golds:
            goldnum += 1
   
        p = tt/prednum if prednum > 0 else 0
        r = tt/goldnum if goldnum > 0 else 0
        f1 = 2*(p*r)/(p+r) if p + r > 0 else 0

        result = {
            'p':p,
            'r':r,
            'f1':f1,
        }
        return result
    

    
