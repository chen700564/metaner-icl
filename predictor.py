import json
import logging
from arguments import ArgumentsForTest
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
import yaml
from utils import *
import tqdm

logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import torch
import tqdm
import os
import numpy as np

def getinstruction(target_labels, mapping, gpt=False):
    if gpt:
        insturction = 'Target entity types: '
        insturction = insturction + ', '.join([mapping[i] for i in target_labels])
        insturction = insturction + '.\n'
    else:
        insturction = ''
        for label in target_labels:
            insturction = insturction + ' <extra_id_7> ' + mapping[label]
    return insturction

def getpredformat(formattext,text):
    return formattext.format(text=text)

def getoutputids(tokenizer, formats,context,targetlabels,mapping):
    outputids = []
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
            index += 1
    outputids = formats['entity']['prefix'] + outputids + formats['entity']['end']
    return outputids

logger = logging.getLogger(__name__)

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        
        newfeatures = []
        for feature in features:
            feature = {
                'input_ids': feature['input_ids'],
                'attention_mask': [1] * len(feature['input_ids'])
            }
            newfeatures.append(feature)

        features = self.tokenizer.pad(
            newfeatures,
            padding=True,
        )
        
        inputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in features.items()}
        return inputs


def gettestset(datapath ,tokenizer, args, formats):
    if type(datapath) is list:
        trainset = datapath[0]
        testset = datapath[1]
        targets = datapath[2]
    else:
        targets = {}
        targetlabels = []
        with open(datapath + '/record.schema') as f:
            for line in f:
                targetlabels.append(json.loads(line))
        for label in targetlabels[0]:
            targets[label] = 'spot'
        with open(datapath + '/train.json') as f:
            trainset = [json.loads(line) for line in f]
        with open(datapath + '/test.json') as f:
            testset = [json.loads(line) for line in f]
    random.shuffle(trainset)
    type2indexs = {i:[] for i in targets}
    labelindex = {i:0 for i in targets}
    for i, instance in enumerate(trainset):
        types = []
        if 'entity' not in instance:
            instance['entity'] = instance['entity_offsets']
        for entity in instance['entity']:
            if entity['type'] in targets:
                types.append(entity['type'])
        for etype in types:
            type2indexs[etype].append(i)
    newtrainset = []
    while len(newtrainset) != len(trainset):
        for label in targets:
            if labelindex[label] >= len(type2indexs[label]):
                continue
            index = labelindex[label]
            instanceindex = type2indexs[label][index]
            if instanceindex not in newtrainset:
                newtrainset.append(instanceindex)
            labelindex[label] += 1
    trainset = [trainset[i] for i in newtrainset]
    nums = len(trainset)
    if nums < args.context_num and args.context_num > 0:
        args.context_num = nums
    instances = []
    mapping = {}
    typeindex = 1
    for label in targets:
        if 'anonymization' in args.enhance:
            mapping[label] = '<type' + str(typeindex) + '>'
        else:
            mapping[label] = label
        typeindex += 1
    targetlabels = targets
    gpt = False
    if args.modeltype != 'metaner':
        gpt = True
    insturction = getinstruction(targets, mapping, gpt)
    insturction = tokenizer.encode(insturction, add_special_tokens=False)

    for context in trainset:
        outputids = []
        if 'entity' not in context:
            context['entity'] = context['entity_offsets']
        context['entity'] = sorted(context['entity'],key=lambda k:k['offset'][0])
        index = 0
        outputids = getoutputids(tokenizer, formats, context,targetlabels,mapping)
        text = getpredformat(formats['entity']['inputformat'],' '.join(context['tokens']))
        textids = tokenizer.encode(text, add_special_tokens=False)
        context['input_ids'] = textids + outputids

    trainset = trainset + trainset
    max_length = 512
    if args.modeltype == 'opt':
        max_length = 2048 - 100
    elif args.modeltype == 'gpt':
        max_length = 1024 - 100

    testindex = 0
    for instance in tqdm.tqdm(testset):
        
        if 'entity' not in instance:
            instance['entity'] = instance['entity_offsets']

        text = getpredformat(formats['entity']['inputformat'],' '.join(instance['tokens']))
        textids = tokenizer.encode(text, add_special_tokens=False)


        
        if args.modeltype != 'metaner':
            textids = textids + formats['entity']['prefix']
            if args.modeltype == 't5' and args.context_num > 0:
                textids = textids + tokenizer.convert_tokens_to_ids(['<extra_id_0>'])
        
        limit = max_length - len(textids)

        usednum = 0
        contextnum = 0
        fullinputids = copy.deepcopy(insturction)
        if args.context_num <= 0:
            newinstance = copy.deepcopy(instance)
            newinstance['index'] = testindex
            newinstance['input_ids'] = fullinputids + textids
            newinstance['targetlabel'] = targets
            if 'anonymization' in args.enhance:
                newinstance['mapping'] = mapping
                newtargets = {}
                for label in targets:
                    newtargets[mapping[label]] = targets[label]
                newinstance['targetlabel'] = newtargets
            instances.append(newinstance)
        else:
            for context in trainset:
                inputids = copy.deepcopy(context['input_ids'])
                if len(inputids) + len(fullinputids) > limit or contextnum == args.context_num:
                    newinstance = copy.deepcopy(instance)
                    newinstance['index'] = testindex
                    newinstance['input_ids'] = fullinputids + textids
                    newinstance['targetlabel'] = targets
                    if 'anonymization' in args.enhance:
                        newinstance['mapping'] = mapping
                        newtargets = {}
                        for label in targets:
                            newtargets[mapping[label]] = targets[label]
                        newinstance['targetlabel'] = newtargets
                    instances.append(newinstance)
                    if usednum > nums:
                        break
                    fullinputids = copy.deepcopy(insturction)
                    contextnum = 0
                fullinputids += inputids
                contextnum += 1
                usednum += 1
        testindex += 1
    return instances    


def predict(model, dataset, data_collator, training_args, args, tokenizer, endid, tag, prefixid = None, textmid = None, enhance = 'None'):
    training_args = copy.deepcopy(training_args)
    training_args.output_dir = training_args.output_dir + '/' + str(tag)

    if model is None:
        if args.modeltype == 'metaner':
            model = AutoModelForSeq2SeqLM.from_pretrained(
                training_args.output_dir,
            )
        elif args.modeltype == 'gpt':
            model = AutoModelForCausalLM.from_pretrained(
                training_args.output_dir,
                pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

    max_length = 512
    if 'opt' in args.modeltype :
        max_length = 2048
    elif args.modeltype == 'gpt':
        max_length = 1024

    model.eval()
    if args.modeltype != 'optbig':
        model = model.cuda()
    decoder_start_token_id = None
    if args.modeltype == 'metaner':
        decoder_start_token_id = prefixid
    elif args.modeltype == 't5':
        endid = tokenizer.eos_token_id
    index = 0
    with torch.no_grad():
        with open(training_args.output_dir + '/' + args.predictfile + '.json', 'w') as f:
            for batch in tqdm.tqdm(data_loader):
                outputs = model.generate(inputs=batch['input_ids'].cuda(),attention_mask=batch['attention_mask'].cuda(),max_length=max_length,num_beams=1,eos_token_id=endid, decoder_start_token_id=decoder_start_token_id,return_dict_in_generate=True,output_scores =True)
                preds = []
                newdataset = []
                batch = outputs.scores[0].size(0)
                for i in range(batch):
                    newdataset.append(dataset[index])
                    index += 1
                for i in range(len(outputs.scores)):
                    score = outputs.scores[i]
                    score, generated = torch.max(score,dim=-1)
                    generated = generated.cpu().numpy()
                    generated = np.expand_dims(generated,1)
                    preds.append(generated)
                preds = np.concatenate(preds,axis=1)
                if args.modeltype == 't5' and args.context_num > 0:
                    preds = preds[:,1:]
                preds, golds, targetlabels, generations = tokenid2result(preds,endid,newdataset,tokenizer)
                for i in range(len(preds)):
                    instance = newdataset[i]
                    f.write(json.dumps({'index':instance['index'], 'generation': generations[i],'gold': golds[i], 'pred': preds[i], 'targetlabels': targetlabels[i]})+'\n')
                    
    return preds

def main():
    parser = HfArgumentParser((
        ArgumentsForTest,
        TrainingArguments
    ))

    args, training_args = parser.parse_args_into_dataclasses()

        

    logger.info("Options:")
    logger.info(args)
    logger.info(training_args)


    if '.yaml' in args.formatsconfig:
        formatsconfig = yaml.load(open(args.formatsconfig),Loader=yaml.FullLoader)
    args.formatsconfig = formatsconfig
    logger.info("context config:")
    logger.info(formatsconfig)

    formats = args.formatsconfig
    args.modeltype = formats['universal']['modeltype']
    args.enhance = formats['universal']['enhance']

    if args.modeltype == 't5' or args.modeltype == 'metaner':
        tokenizer = AutoTokenizer.from_pretrained(args.plm)
    elif args.modeltype == 'gpt':
        tokenizer = AutoTokenizer.from_pretrained(args.plm,padding_side = "left",add_prefix_space=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif 'opt' in args.modeltype:
        tokenizer = AutoTokenizer.from_pretrained(args.plm,padding_side = "left", use_fast=False)



    
    for task in formats:
        for tag in formats[task]:
            if 'format' not in tag:
                formats[task][tag] = tokenizer.encode(formats[task][tag], add_special_tokens=False)
    

    if args.debugfile != 'None':
        seeds = os.listdir(args.testset)
        for seed in seeds:
            datasetpath = args.testset + '/' + seed + '/' + str(args.shot_num) + 'shot'
            testset = gettestset(datasetpath ,tokenizer, args, formats)
            index = 0
            f = open(args.debugfile+'_'+str(seed),'w')
            for instance in tqdm.tqdm(testset):
                text = tokenizer.decode(instance['input_ids'])
                f.write(json.dumps({'index': instance['index'], 'text': text, 'entity': instance['entity']}))
                # f.write(tokenizer.decode(instance['input_ids'])+'\n')




    data_collator = DataCollator(tokenizer=tokenizer)

    if args.context_num >= 0:
        if args.modeltype == 't5' or args.modeltype == 'metaner':
            logger.info("seq2seq model")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.plm,decoder_start_token_id = formats['universal']['prefix'][0],
            )
        elif args.modeltype == 'gpt':
            logger.info("gpt model")
            model = AutoModelForCausalLM.from_pretrained(
                args.plm,
                pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.float16
            )
        elif args.modeltype == 'optbig':
            logger.info("opt big model")
            import nvgpu
            def max_memo():
                gpus = nvgpu.available_gpus()
                max_memo = {}
                for gpu in gpus:
                    gpu = int(gpu)
                    max_memo[gpu] = "68000MiB"
                return max_memo
            max_memo = max_memo()
            model = AutoModelForCausalLM.from_pretrained(args.plm, torch_dtype=torch.float16, device_map="auto", max_memory=max_memo)
        elif args.modeltype == 'opt':
            logger.info("opt model")
            model = AutoModelForCausalLM.from_pretrained(
                    args.plm, torch_dtype=torch.float16
                )
    else:
        model = None

    seeds = os.listdir(args.testset)
    for seed in seeds:
        set_seed(args.randomseed)
        datasetpath = args.testset + '/' + seed + '/' + str(args.shot_num) + 'shot'
        testset = gettestset(datasetpath ,tokenizer, args, formats)
        endid = formats['universal']['end'][0] if args.modeltype == 't5' or args.modeltype == 'metaner' else formats['universal']['end'][-1]
        predict(model, testset, data_collator, training_args, args, tokenizer, endid, seed, prefixid = formats['universal']['prefix'][0], textmid=[formats['entity']['context_left'],formats['entity']['context_mid']],enhance=args.enhance)

if __name__ == "__main__":
    main()