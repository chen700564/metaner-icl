from typing import List
import itertools
import numpy as np
import copy

def tokenid2result(predictions,endid,dataset,tokenizer, enhance = 'None'):
    generations = []
    preds = []
    golds = []
    targetlabels = []
    for index in range(len(predictions)):
        try:
            tokenids = predictions[index].tolist()
        except:
            tokenids = predictions[index]
        endindex = len(tokenids)
        tokenids = [i if i >= 0 else endid for i in tokenids]
        if endid in tokenids:
            endindex = tokenids.index(endid)
        tokenids = tokenids[:endindex]
        # print(tokenizer.decode(tokenids))
        try:
            text = tokenizer.decode(tokenids)
        except:
            print(tokenids)
        
        instance = dataset[index]
        mapping = None
        if 'anonymization' in enhance:
            mapping = {}
            for label in instance['targetlabel']:
                if instance['targetlabel'][label] == 'spot':
                    mapping['entity'] = label
        if 'is<' in text:
            text = text.replace('is<', 'is <')
        generations.append(text)
        targetlabel = instance['targetlabel']
        if 'mapping' in instance:
            mapping = instance['mapping']
            newmapping = {}
            for label in mapping:
                newmapping[mapping[label]] = label
            targetlabel = {}
            for label in instance['targetlabel']:
                if label in newmapping:
                    targetlabel[newmapping[label]] = instance['targetlabel'][label]
            mapping = newmapping
        pred = decode(tokenizer,instance['tokens'],text,targetlabel, mapping)
        preds.append(pred)
        gold = []
        typename = 'entity'
        if typename not in instance:
            typename = 'entity_offsets'
        for entity in instance[typename]:
            if entity['type'] in targetlabel:
                gold.append(entity)
        golds.append(gold)
        targetlabels.append(targetlabel)
    return preds, golds, targetlabels, generations

def tempdecode(text):
    indexs = [i for i in range(len(text)) if text.startswith('.', i)]
    start = 0
    preds = []
    prefixtext = None
    for i in indexs:
        subtext = text[start:i].split(' ')

        if 'is' in subtext and subtext[-1] != 'is':
            indexs2 =  [i for i,a in enumerate(subtext) if a=='is']
            index = indexs2[-1]
            span1 = ' '.join(subtext[:index]).strip()
            asoc = subtext[index+1:]
            preds.append([span1,' '.join(asoc),''])
            if ',' in ' '.join(asoc):
                types = ' '.join(asoc).split(',')
                types = [i.strip() for i in types]
                preds.append([span1,types,''])
            start = i + 1
            if prefixtext is not None:
                lastpred = copy.deepcopy(preds[-1])
                lastpred[0] = ' '.join(subtext).strip() + ' . ' + lastpred[0] 
                preds.append(lastpred)

                prefixtext = None
        # the rest of span will be merged into last preds
        elif len(preds) > 0:
            lastpred = copy.deepcopy(preds[-1])
            lastpred[0] = lastpred[0] + ' . ' + ' '.join(subtext).strip()
            preds.append(lastpred)
            prefixtext = subtext
    return preds


    
def decode(tokenizer,oritokens,generated,targets,mapping=None):
    tokens = [tokenizer.decode(tokenizer.encode(token,add_special_tokens=False)).replace(' ','') for token in oritokens]
    tokenindex = []
    l = 0
    for token in tokens:
        for i in range(len(token)):
            tokenindex.append(l)
        l += 1
    tokenindex.append(l)
    tokens = ''.join(tokens) 
    results = tempdecode(generated)
    preds = []
    newresult = []
    for result in results:
        flag = False
        if type(result[1]) is list:
            newpreds = []
            for etype in result[1]:
                if mapping is not None and etype in mapping:
                    newpreds.append(mapping[etype])
                else:
                    newpreds.append(etype)
            newpreds = [i for i in newpreds if i in targets]
            if len(newpreds) > 0:
                result[1] = newpreds
                flag = True
        else:
            if mapping is not None and result[1] in mapping:
                result[1] = mapping[result[1]]
            elif mapping is not None and result[1] + ' of ' + result[0] in mapping:
                result[1] = mapping[result[1] + ' of ' + result[0]]
            elif mapping is not None and result[1] + ' of ' + result[0] in targets:
                result[1] = result[1] + ' of ' + result[0]
                result[0] = result[2]
                result[2] = ''
            if result[1] in targets:
                flag = True
        if flag:
            newresult.append(result)
    results = newresult
    for result in results:
        # print(result)
        span1 = result[0].replace(' ','')
        span2 = result[2].replace(' ','')
        label = result[1]
        if len(span1) > 0:
            offsets = findoffset(span1,tokens,tokenindex)
            if len(offsets) > 0:
                preds.append(['spot',label,offsets,' '.join(oritokens[offsets[0][0]:offsets[0][1]])])
    preds = dealconflict(preds)             
    return getresult(preds)

def getresult(preds):
    final = []
    for pred in preds:
        if type(pred[1]) is list:
            for etype in pred[1]:
                final.append({"type":etype,'offset':list(range(pred[2][0],pred[2][1])),'text':pred[3]})
        else:
            final.append({"type":pred[1],'offset':list(range(pred[2][0],pred[2][1])),'text':pred[3]})
    return final

def dealconflict(preds):
    '''
    longest first
    '''
    final = []
    used_offsets = []
    lengths = []
    for pred in preds:
        lengths.append(pred[2][0][1]-pred[2][0][0])
    lengths = np.array(lengths)
    indexs = np.argsort(-lengths)
    for oindex in indexs:
        offsets = preds[oindex][2]
        for offset in offsets:
            flag = True
            for index in range(offset[0],offset[1]):
                if index in used_offsets:
                    flag = False
                    break
            if flag:
                final.append([preds[oindex][0],preds[oindex][1],offset,preds[oindex][3]])
                for index in range(offset[0],offset[1]):
                    if index not in used_offsets:
                        used_offsets.append(index)
    return final
            
            

def findoffset(text,rawtext,tokenindex):
    start = 0
    text = text.replace(' ','')
    offsets = []
    while True:
        if text in rawtext[start:] and len(text) > 0:
            index = rawtext[start:].index(text)
            offset = [tokenindex[start + index],tokenindex[start + index + len(text)]]
            if (start + index == 0 or tokenindex[start + index-1] != tokenindex[start + index]) and (start + index + len(text) >= 1 and tokenindex[start + index-1 + len(text) ] != tokenindex[start + index + len(text) ]):
                offsets.append(offset)
            start =  start + index + len(text)
        else:
            break
    return offsets
