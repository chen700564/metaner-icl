import json
import logging
import random
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers import Trainer, Seq2SeqTrainer
import yaml
import sys
import gc
import torch
import tqdm


logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Optional
import copy
from pretrain_utlis import *

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def labelpad(self, labels, features):

        if 'decoder_input_ids' not in features:
            sequence_length = len(features["input_ids"][0])
        else:
            sequence_length = len(features["decoder_input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            features["labels"] = [
                list(label) + [-100] * (sequence_length - len(label)) for label in labels
            ]
        else:
            features["labels"] = [
                [-100] * (sequence_length - len(label)) + list(label) for label in labels
            ]
        return features

    def getpad(self,features,labels=None):
        features = self.tokenizer.pad(
                features,
                padding=True,
            )
        if 'decoder_input_ids' in features:
            features2 = self.tokenizer.pad(
                    {'input_ids': features['decoder_input_ids'], 'attention_mask': features['decoder_attention_mask']},
                    padding=True
            )
            features['decoder_input_ids'] = features2['input_ids']
            features['decoder_attention_mask'] = features2['attention_mask']
        if labels is not None:
            features = self.labelpad(labels,features)
        return features

    def __call__(self, features):

        rawinputs = []
        rawlabels = []
        instanceindex = 0
        for feature in features:
            rawfeature = {}
            rawfeature['input_ids'] = feature['input_ids']
            rawfeature['attention_mask'] = feature['attention_mask']
            if 'decoder_input_ids' in feature:
                rawlabels.append(feature['labels'][1:])
                rawfeature['decoder_input_ids'] = feature['decoder_input_ids'][:-1]
                rawfeature['decoder_attention_mask'] = [1] * len(rawfeature['decoder_input_ids'])
            else:
                rawlabels.append(feature['labels'])
            rawinputs.append(rawfeature)

            instanceindex += 1
        rawinputs = self.getpad(rawinputs,labels=rawlabels)
        rawinputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in rawinputs.items()}
        inputs = rawinputs
        return inputs

def getinstruction(target_labels, mapping, modeltype):
    if modeltype == 't5':
        insturction = ''
        for label in target_labels:
            insturction = insturction + ' <extra_id_7> ' + mapping[label]
    elif modeltype == 'gpt':
        insturction = 'Target entity types: '
        insturction = insturction + ', '.join([mapping[i] for i in target_labels])
        insturction = insturction + '.\n'
    return insturction



class Seq2seqDataset(Dataset):
    def __init__(self, dataset, targetlabels, tokenizer, formats, endid, modeltype, min_labels= None):
        super(Seq2seqDataset).__init__()
        self.targetlabels = targetlabels
        if dataset is not list:
            self.dataset = []
            with open(dataset) as f:
                for line in f:
                    instance = json.loads(line)
                    if 'entity' not in instance:
                        instance['entity'] = instance['entity_offsets']
                    instance['entity'] = sorted(instance['entity'],key=lambda i: i['offset'][0]) 
                    self.dataset.append(instance)
                    # reject
                    for i in range(5):
                        newinstance = self.getnegdata(instance)
                        if newinstance is not None:
                            self.dataset.append(newinstance)
        else:
            self.dataset = dataset
        
        max_labels = len(targetlabels)
        self.tokenizer = tokenizer
        self.formats = formats
        self.endid = endid
        self.modeltype = modeltype

        if min_labels is None or max_labels is None or min_labels > max_labels:
            self.min_labels = len(targetlabels)
            self.max_labels = len(targetlabels)
        else:
            self.min_labels = min(len(targetlabels),min_labels)
            self.max_labels = min(len(targetlabels),max_labels)
        
    def getoutputids(self, tokenizer, formats,context,targetlabels,mapping):
        outputids = []
        labels = []
        index = 0
        for entity in context['entity']:
            type = entity['type']
            if type in targetlabels:
                text = entity['text'] + ' is ' + mapping[type] + '.'
                if index == 0:
                    outputid = tokenizer.encode(text, add_special_tokens=False)
                else:
                    outputid = tokenizer.encode(' '+text, add_special_tokens=False)
                outputids += outputid
                labels += outputid
                index += 1
            else:
                text = entity['text'] + ' is'
                if index == 0:
                    outputid = tokenizer.encode(text, add_special_tokens=False)
                else:
                    outputid = tokenizer.encode(' '+text, add_special_tokens=False)
                labelid = tokenizer.encode(' not entity.', add_special_tokens=False)
                outputids += (outputid + labelid)
                labels += ([-100] * len(outputid) + labelid)
                index += 1 

        outputids = formats['entity']['prefix'] + outputids + formats['entity']['end']
        labels = formats['entity']['prefix'] + labels + formats['entity']['end']
        return outputids, labels
    
    def getnegdata(self,data):
        data = copy.deepcopy(data)
        length = [1,2,3]
        label = np.array([0] * len(data['tokens']))
        for i in range(len(data['tokens'])):
            if not data['tokens'][i].isalnum():
                label[i] = 1
        for entity in data['entity']:
            if self.targetlabels is None or entity['type'] in self.targetlabels:
                for i in entity['offset']:
                    label[i] = 1
        length = random.choice(length)
        index = np.where(label == 0)[0]
        if len(index) > 0:
            start = np.random.choice(index)
            end = start + 1
            for i in range(length):
                if end < len(label) and label[end] == 0:
                    end += 1
                else:
                    break
            newentity = {
                'type':'NOTENT',
                'text': ' '.join(data['tokens'][start:end]),
                'offset': list(range(start,end)),
            }
            data['entity'].append(newentity)
            data['entity'] = sorted(data['entity'],key=lambda i: i['offset'][0])
            return data
        else:
            return None

    def getinstance(self,instance):
        
        instance = copy.deepcopy(instance)

        if self.min_labels < len(self.targetlabels):

            num_targetlabels = random.choice(range(self.min_labels,self.max_labels+1))
            sampledlabels = random.sample(self.targetlabels,num_targetlabels)

            newentitiess = []
            for entity in instance['entity']:
                if entity['type'] in sampledlabels or entity['type'] == 'NOTENT':
                    newentitiess.append(entity)
            instance['entity'] = newentitiess
        else:
            sampledlabels = self.targetlabels
        random.shuffle(sampledlabels)
        mapping = {i:i for i in sampledlabels}
        instance = getdata(self.tokenizer, instance, 200)
        text = ' '.join(instance['tokens'])
        text = getpredformat(self.formats['entity']['inputformat'],text)
        inputids = self.tokenizer.encode(text, add_special_tokens=False)
        # inputids = self.tokenizer.encode(text, add_special_tokens = False) + [self.endid]

        instruction = getinstruction(sampledlabels, mapping, self.modeltype)
        instructionid = self.tokenizer.encode(instruction, add_special_tokens=False)
        outputs, labels = self.getoutputids(self.tokenizer, self.formats, instance,sampledlabels,mapping)

        inputids = instructionid + inputids

        if self.modeltype == 't5':
            instance = {
                'input_ids': inputids,
                'attention_mask': [1] * len(inputids),
                'decoder_input_ids': outputs,
                'labels': labels,
            }
        elif self.modeltype == 'gpt':
            labels = [-100] * len(inputids) + labels
            inputids = inputids + outputs
            instance = {
                'input_ids': inputids,
                'attention_mask': [1] * len(inputids),
                'labels': labels,
            }
        return instance

    def __getitem__(self, index):
        return self.getinstance(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)


@dataclass
class Arguments:

    plm: str = field(
        metadata={
            "help": "Pretrained model"
        },
    )
    dataset: Optional[str] = field(
        default="entity/conll03", metadata={"help": "dataset path"}
    )
    formatsconfig: Optional[str] = field(
        default="config/formats/finetune/t5.yaml", metadata={"help": "config file for ict"}
    )
    randomseed: Optional[int] = field(
        default=2333, metadata={"help": "random seed"}
    )
    shot: Optional[int] = field(
        default=5, metadata={"help": "shot number"}
    )

    input_maxlength: Optional[int] = field(
        default=100, metadata={"help": "max token id length for input instance"}
    )
    predictfile: Optional[str] = field(
        default="prediction", metadata={"help": "name of predict file"}
    )
    debugfile: Optional[str] = field(
        default="None", metadata={"help": "debugfile"}
    )



def modelfinetune(args, training_args, tokennum, tokenizer, data_collator, trainset, tag):
    if args.modeltype == 't5':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.plm,
        )
    elif args.modeltype == 'gpt':
        model = GPT2LMHeadModel.from_pretrained(
            args.plm,
        )
    if tokennum is not None:
        model.resize_token_embeddings(tokennum)

    training_args = copy.deepcopy(training_args)
    training_args.output_dir = training_args.output_dir + '/' + str(tag)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()
    del model
    gc.collect()
    torch.cuda.empty_cache()



def main():
    parser = HfArgumentParser((
        Arguments,
        TrainingArguments
    ))

    args, training_args = parser.parse_args_into_dataclasses()

    args.training = training_args.do_train
    args.testing = training_args.do_predict
        

    logger.info("Options:")
    logger.info(args)
    logger.info(training_args)

    if '.yaml' in args.formatsconfig:
        formatsconfig = yaml.load(open(args.formatsconfig),Loader=yaml.FullLoader)

    args.modeltype = formatsconfig['universal']['modeltype']


    set_seed(args.randomseed)
    tokennum = None
    if args.modeltype == 't5':
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
        if 'additional_special_tokens' not in tokenizer.special_tokens_map or '<extra_id_0>' not in tokenizer.special_tokens_map['additional_special_tokens']:
            specicaltokens = {
                'additional_special_tokens': ['<extra_id_0>','<extra_id_1>','<extra_id_2>','<extra_id_3>','<extra_id_4>','<extra_id_5>','<extra_id_6>','<extra_id_7>','<extra_id_8>']
            }

            tokenizer.add_special_tokens(specicaltokens)
            tokennum = len(tokenizer)
    elif args.modeltype == 'gpt':
        tokenizer = AutoTokenizer.from_pretrained(args.plm,add_prefix_space=True)
        tokenizer.pad_token = tokenizer.eos_token


            
    logger.info("context config:")
    logger.info(formatsconfig)
    args.formatsconfig = formatsconfig

    seeds = os.listdir(args.dataset)
    formats = args.formatsconfig
    for task in formats:
        for tag in formats[task]:
            if 'format' not in tag:
                formats[task][tag] = tokenizer.encode(formats[task][tag], add_special_tokens=False)
    endid = formats['universal']['end'][0] if args.modeltype == 't5' else formats['universal']['end'][-1]

    if args.debugfile != 'None':
        seed = 'seed1'
    
        targetlabels = []
        datasetpath = args.dataset + '/' + seed + '/' + str(args.shot) + 'shot'
        with open(datasetpath + '/record.schema') as f:
            for line in f:
                targetlabels.append(json.loads(line))

        targets = targetlabels[0]
        data_collator = DataCollator(tokenizer)
        trainset = Seq2seqDataset(datasetpath+'/train.json',targets,tokenizer,formats,endid,args.modeltype,3)
        dataloader = DataLoader(trainset,batch_size=1,collate_fn=data_collator)
        with open(args.debugfile,'w') as f:
            for index,batch in tqdm.tqdm(enumerate(dataloader)):
                inputids = batch['input_ids'].tolist()
                f.write('raw_inputs\n')
                inputids = inputids[0]
                f.write(tokenizer.decode(inputids)+'\n')
                decoder_input_ids = batch['decoder_input_ids'].tolist()
                f.write('raw_output\n')
                f.write(tokenizer.decode(decoder_input_ids[0])+'\n')

    for seed in seeds:
        targetlabels = []
        datasetpath = args.dataset + '/' + seed + '/' + str(args.shot) + 'shot'
        with open(datasetpath + '/record.schema') as f:
            for line in f:
                targetlabels.append(json.loads(line))

        targets = targetlabels[0]
        

        data_collator = DataCollator(tokenizer)
        trainset = Seq2seqDataset(datasetpath+'/train.json',targets,tokenizer,formats,endid,args.modeltype,3)
        modelfinetune(args, training_args, tokennum, tokenizer, data_collator, trainset, seed)


if __name__ == "__main__":
    main()