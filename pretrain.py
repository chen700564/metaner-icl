from torch.utils.data import Dataset,IterableDataset
from torch.optim import AdamW
import json
import os
import copy
import random
import tqdm
import logging
import torch
from trainer import MetaTrainer
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
        iclinputs = []
        queryinputs = []
        indexs = []
        iclindex = []
        instanceindex = 0
        dev = False
        for feature in features:
            rawfeature = {}
            rawfeature['input_ids'] = feature['input_ids']
            rawfeature['attention_mask'] = feature['attention_mask']
            if 'dev' in feature:
                rawinputs.append(rawfeature)
                dev = True
                continue
            if 'decoder_input_ids' in feature:
                rawlabels.append(feature['labels'][1:])
                rawfeature['decoder_input_ids'] = feature['decoder_input_ids'][:-1]
                rawfeature['decoder_attention_mask'] = [1] * len(rawfeature['decoder_input_ids'])
                rawinputs.append(rawfeature)
            if 'indexs' in feature and len(feature['indexs']) > 0:
                iclfeature = {
                    'input_ids': feature['icl_input_ids'],
                    'attention_mask': feature['icl_attention_mask'],
                }
                if 'icl_decoder_input_ids' in feature:
                    iclfeature['decoder_input_ids'] = [i[:-1] for i in feature['icl_decoder_input_ids']]
                    iclfeature['decoder_attention_mask'] = [[1] * (len(i) - 1) for i in feature['icl_decoder_input_ids']]
                    icllabels = [i[1:] for i in feature['icl_decoder_input_ids']]
                else:
                    icllabels = feature['icl_input_ids']
                iclfeature = self.getpad(iclfeature,labels=icllabels)
                iclfeature = {k:torch.tensor(v, dtype=torch.int64) for k,v in iclfeature.items()}
                iclinputs.append(iclfeature)
                queryfeature = {
                    'input_ids': [feature['query_input_ids']],
                    'attention_mask': [[1] * len(feature['query_input_ids'])],
                }
                queryfeature = {k:torch.tensor(v, dtype=torch.int64) for k,v in queryfeature.items()}
                queryinputs.append(queryfeature)
                indexs.append(feature['indexs'])
                iclindex.append(instanceindex)
            instanceindex += 1
        
        if dev:
            rawinputs = self.getpad(rawinputs,labels=None)
            rawinputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in rawinputs.items()}
            inputs = rawinputs
        else:
            rawinputs = self.getpad(rawinputs,labels=rawlabels)
            rawinputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in rawinputs.items()}
            inputs = rawinputs
            inputs['icl'] = iclinputs
            inputs['query'] = queryinputs
            inputs['indexs'] = indexs
            inputs['iclindex'] = iclindex
        return inputs

class fulldevdataset(Dataset):
    def __init__(self, datasets) -> None:
        super(fulldevdataset).__init__()
        self.datasets = []
        for dataset in datasets:
            for instance in dataset:
                instance['indexs'] = []
                self.datasets.append(instance)
    
    def __getitem__(self, index):
        return self.datasets[index]
    
    def __len__(self):
        return len(self.datasets)

class fulldataset(IterableDataset):
    def __init__(self, datasets) -> None:
        super(fulldataset).__init__()
        self.datasets = datasets
        self.lengths = []
        for dataset in datasets:
            try:
                len(dataset)
                self.lengths.append(1)
            except:
                self.lengths.append(5)
    
    def readdataset(self,dataset):
        while True:
            for data in dataset:
                yield data
    
    def __iter__(self):
        datas = [self.readdataset(i) for i in self.datasets]
        
        while True:   
            index = 0
            for dataset in datas:
                for i in range(self.lengths[index]):
                    yield next(dataset)
                index += 1

class devDataset(Dataset):
    def __init__(self, files, contextnum, tokenizer, max_instance_length, formats, trainset,enhance='None', endid = None, labels=None):
        super(devDataset).__init__()
        self.contextnum = contextnum
        self.files = files
        self.max_instance_length = max_instance_length
        self.tokenizer = tokenizer
        self.formats = formats
        self.enhance = enhance
        self.labels = labels
        self.endid = endid
        self.dataset = []
        for file in files:
            with open(file) as f:
                for line in tqdm.tqdm(f):
                    line = json.loads(line)
                    line = trainset.getinstance(line)                    
                    line['indexs'] = []
                    line['dev'] = True
                    self.dataset.append(line)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class pretrainDataset(IterableDataset):
    def __init__(self, files, contextnum, tokenizer, max_instance_length, formats,enhance='None', lmfile = None, endid = None, labels=None, code2name=None):
        super(pretrainDataset).__init__()
        self.contextnum = contextnum
        self.files = files
        self.max_instance_length = max_instance_length
        self.tokenizer = tokenizer
        self.formats = formats
        self.enhance = enhance
        self.lmfile = lmfile
        self.labels = labels
        self.endid = endid
        self.code2name = code2name
    
    def getnerinstance(self, instance, labels, lm = False):
        instance = getnerinstance(self.tokenizer, self.formats, instance, labels, self.endid, lm, self.code2name)
        return instance

    def sampledata(self, instance):
        contexts = instance['instances']
        labels = list(instance['instances'].keys())
        targetlabels = random.sample(labels,min(5,len(labels)))
        index2instance = instance['full_instances']
        contextsindexs = sample_contexts(self.contextnum,contexts,targetlabels)

        candiindexs = []
        if random.random() > 0.4:
            '''
            sample more positive instances
            '''
            candiindexs = []
            for label in targetlabels:
                candiindexs = candiindexs + [i for i in instance['instances'][label] if i not in contextsindexs and i not in candiindexs] 
        if len(candiindexs) == 0:
            candiindexs = [i for i in instance['fullindexs'] if i not in contextsindexs]  
            
        instanceindex = random.choice(candiindexs)

        if type(index2instance) is not list:
            instanceindex = str(instanceindex)
        
        
        queryinstance = index2instance[instanceindex]
        queryinstance = getdata(self.tokenizer, queryinstance,100)

        if type(index2instance) is not list:
            contexts = [index2instance[str(i)] for i in contextsindexs]
        else:
            contexts = [index2instance[i] for i in contextsindexs]
        
        noveltypes = []
        for context in contexts[:5]:
            if type(context['entity'][0]['type']) is list:
                flag = True
                for entity in context['entity']:
                    for label in entity['type']:
                        if label not in targetlabels and label not in noveltypes:
                            noveltypes.append(label)
                if flag:
                    noveltypes = noveltypes + entity['type']
            else:
                noveltypes = [i['type'] for i in context['entity'] if i['type'] not in targetlabels]

        maxnoveltypes = random.choice(range(0,10))
        noveltypes = random.sample(noveltypes,min(len(noveltypes),maxnoveltypes))
        targetlabels = targetlabels + noveltypes
        return queryinstance, contexts, targetlabels

    def getinstance(self, instance, lm= False):
        queryinstance, contexts, targetlabels = self.sampledata(instance)
        instance = getinstance(queryinstance, contexts, self.tokenizer, self.max_instance_length, self.formats, targetlabels, self.enhance, lm, self.code2name)
        return instance

    def readfile(self,file):
        while True:
            with open(file) as f:
                for line in f:
                    line = json.loads(line)
                    if 'instances' in line and len(line['instances']) == 0:
                        continue
                    yield line
    
    def readnerfile(self, file):
        while True:
            with open(file) as f:
                for line in f:
                    line = json.loads(line)
                    if 'full_instances' in line and len(line['full_instances']) == 0:
                        continue
                    for index in line['full_instances']:
                        yield line['full_instances'][index]
    
    def readlmfile(self,file):
        while True:
            with open(file) as f:
                for line in f:
                    line = json.loads(line)
                    labels = list(line['instances'].keys())
                    if len(labels) == 0:
                        continue
                    for instance in line['full_instances']:
                        yield instance, labels

    def __iter__(self):    
        f1 = self.readfile(self.files[0])
        if 'nerpretrain' in self.enhance:
            f2 = self.readnerfile(self.files[0])
        if 'lmpretrain' in self.enhance:
            f3 = self.readfile(self.lmfile)
            f4 = self.readlmfile(self.lmfile)
        while True:
            num = 4
            for i in range(num):
                yield self.getinstance(next(f1))
                if 'nerpretrain' in self.enhance:
                    yield self.getnerinstance(next(f2),self.labels)
            if 'lmpretrain' in self.enhance:
                yield self.getinstance(next(f3), lm = True)
                instance, labels = next(f4)
                yield self.getnerinstance(instance, labels, lm = True)

def preparedataset(file):
    label2index = {}
    dataset = []
    with open(file) as f:
        index = 0
        for line in tqdm.tqdm(f):
            line = json.loads(line)
            instance = {
                'tokens': line['tokens'],
                'entity': []
            }
            entityname = 'entity'
            if 'entity' not in line:
                entityname = 'entity_offsets'
            
            # sorted instance
            entities = line[entityname]
            entities = sorted(entities,key=lambda k:k['offset'][0])

            instance['entity'] = entities

            types = []
            for entity in entities:
                if entity['type'] not in types:
                    types.append(entity['type'])

            if len(types) == 0:
                types = ['N/A']

            for label in types:
                if label not in label2index:
                    label2index[label] = []
                label2index[label].append(index)
            dataset.append(instance)
            index += 1
    return dataset, label2index

def datareader(option,tokenizer,formats,debug=False):
    '''
    read data from option.dataset
    '''
    
    if option.training:
        max_instance_length = 512
        endid = formats['universal']['end'][0]
        trainsets = []
        valsets = []
        datapath = option.dataset
        files = [datapath + '/ICL_train.json']
        labels = None
        if os.path.exists(datapath + '/code2name.json'):
            code2name = json.load(open(datapath + '/code2name.json'))
        else:
            code2name = None
        if 'lmpretrain' in option.enhance or 'nerpretrain' in option.enhance:
            labels = None
            nerpath = datapath
            lmpath = datapath
            if 'nerpretrain' in option.enhance:
                label2id = json.load(open(nerpath + '/label2id.json'))
                labels = [i for i in label2id]
            trainset = pretrainDataset(files,option.context_num,tokenizer,max_instance_length, formats, enhance=option.enhance,lmfile=lmpath + '/lmtrain.json', endid=endid,labels=labels,code2name=code2name)
        else:
            trainset = pretrainDataset(files,option.context_num,tokenizer,max_instance_length, formats,enhance=option.enhance,code2name=code2name)
        devfile = datapath + '/ICL_dev.json'
        if not debug and os.path.exists(devfile):
            valset = devDataset([devfile],option.context_num,tokenizer,max_instance_length, formats, trainset, enhance = option.enhance, labels = labels)
        else:
            valset = None
        trainsets.append(trainset)
        if valset is not None:
            valsets.append(valset)
        trainset = fulldataset(trainsets)
        if len(valsets) > 0:
            valset = fulldevdataset(valsets)
        else:
            valset = None
        return trainset,valset


import json
import logging
from arguments import Arguments
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers import Trainer
import yaml
import tqdm
import sys

logging.basicConfig(level = logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((
        Arguments,
        TrainingArguments
    ))

    args, training_args = parser.parse_args_into_dataclasses()

    args.training = training_args.do_train
    args.testing = training_args.do_predict

    args.modelpath = training_args.output_dir
        

    logger.info("Options:")
    logger.info(args)
    logger.info(training_args)


    set_seed(args.randomseed)

    tokenizer = AutoTokenizer.from_pretrained(args.plm)

    labelnames = []
    if 'anonymization' in args.enhance:
        labelnames = ['<type' + str(i) + '>' for i in range(1, 99)]
    if 'lmpretrain' in args.enhance:
        masknames = ['<mask' + str(i) + '>' for i in range(1, 99)]
        labelnames = labelnames + masknames
    specicaltokens = {
        'additional_special_tokens': labelnames
    }
    tokenizer.add_special_tokens(specicaltokens)
    tokennum = len(tokenizer)


    if '.yaml' in args.datasetconfig:
        datasetconfig = yaml.load(open(args.datasetconfig),Loader=yaml.FullLoader)
    args.datasetconfig = datasetconfig

    if '.yaml' in args.formatsconfig:
        formatsconfig = yaml.load(open(args.formatsconfig),Loader=yaml.FullLoader)
    logger.info("context config:")
    logger.info(formatsconfig)
    args.formatsconfig = formatsconfig

    formats = args.formatsconfig
    for task in formats:
        for tag in formats[task]:
            if 'format' not in tag:
                formats[task][tag] = tokenizer.encode(formats[task][tag], add_special_tokens=False)
    
    debug = False
    if args.debugfile != 'None':
        debug = True
    trainset,valset = datareader(args,tokenizer,formats,debug)
    if valset is None:
        training_args.do_eval = False
    

    if args.debugfile != 'None':
        with open(args.debugfile,'w') as f:
            data_collator = DataCollator(tokenizer=tokenizer)
            dataloader = DataLoader(trainset,batch_size=1,collate_fn=data_collator)
            for index,batch in tqdm.tqdm(enumerate(dataloader)):
                inputids = batch['input_ids'].tolist()
                f.write('raw_inputs\n')
                inputids = inputids[0]
                f.write(tokenizer.decode(inputids)+'\n')
                if 'decoder_input_ids' in batch:
                    decoder_inputids = batch['decoder_input_ids'].tolist()
                    f.write('raw_decoder_inputs\n')
                    decoder_inputids = decoder_inputids[0]
                    f.write(tokenizer.decode(decoder_inputids)+'\n')
                labels = batch['labels'].tolist()
                label = [i if i >=0 else 0 for i in labels[0]]
                f.write('raw_output\n')
                f.write(tokenizer.decode(label)+'\n')
        raise

    data_collator = DataCollator(tokenizer=tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.plm,
    )
    if tokennum is not None:
        model.resize_token_embeddings(tokennum)
    

    model = model.cuda()

    training_args.endid = formats['universal']['end'][0]
    training_args.prefixid = formats['universal']['prefix'][0]

    trainer = MetaTrainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    trainer.train()


if __name__ == "__main__":
    main()