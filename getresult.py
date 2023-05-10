import json
import argparse
import os
import numpy as np
import sys
import copy

def writeresult(modelfile,resultname,f1,wrongmap,errorinfo,errors=None):
    filename = ['f1','error','errorinfo','wrongmap']
    filename = ['/' + resultname + '_' + i for i in filename]
    json.dump(f1,open(modelfile+filename[0]+'.json','w'))
    json.dump(wrongmap,open(modelfile+filename[3]+'.json','w'))
    json.dump(errorinfo,open(modelfile+filename[2]+'.json','w'))
    if errors is not None:
        with open(modelfile+filename[1]+'.txt','w') as f:
            for e in errors:
                if len(e['wrongspan']) > 0 or len(e['unpred']) > 0 or len(e['wronglabel']) > 0 or len(e['wrongmargin']) > 0 or len(e['allspanerror']) > 0:
                    f.write(str(e['index'])+'\n')
                    f.write('[Gold]:'+str(e['gold'])+'\n')
                    f.write('[Pred]:'+str(e['pred'])+'\n')
                    if len(e['unpred']) > 0:
                        f.write('[Unpredicted]:'+str(e['unpred'])+'\n')
                    if len(e['wrongspan']) > 0:
                        f.write('[wrongSpan]:'+str(e['wrongspan'])+'\n')
                    if len(e['wronglabel']) > 0:
                        f.write('[wrongLabel]:'+str(e['wronglabel'])+'\n')
                    if len(e['wrongmargin']) > 0:
                        f.write('[wrongMargin]:'+str(e['wrongmargin'])+'\n') 
                    if len(e['allspanerror']) > 0:
                        f.write('[otherspan]:'+str(e['allspanerror'])+'\n') 

def geterror(pred,gold,length,wrongmap):
    '''
    margin error: one of the token in span is true 
    label error: span is correct, label is wrong
    all span error: all span should be other
    span error: one of the token in span is other label 
    '''
    predlabel = [0] * length
    goldlabel = [0] * length
    wrongSpan = []
    wrongmargin = []
    allSpanerror = []
    unpred = []
    for entity in pred:
        if 'offset' not in entity:
            return wrongSpan,wrongmargin,unpred,allSpanerror,wrongmap

        for i in entity['offset']:
            predlabel[i] = entity['type']
    for entity in gold:
        if 'offset' not in entity:
            return wrongSpan,wrongmargin,unpred,allSpanerror,wrongmap
        for i in entity['offset']:
            goldlabel[i] = entity['type']
    for entity in pred:
        if entity not in gold:
            type = entity['type']
            if type not in wrongmap:
                wrongmap[type] = {}
            error = 'allspan'
            for i in entity['offset']:
                if goldlabel[i] == type:
                    error = 'margin'
                    if 'margin' not in wrongmap[type]:
                        wrongmap[type]['margin'] = 1
                    else:
                        wrongmap[type]['margin'] += 1
                    break
                elif goldlabel[i] != 0:
                    error = 'span'
                    if goldlabel[i] not in wrongmap[type]:
                        wrongmap[type][goldlabel[i]] = 1
                    else:
                        wrongmap[type][goldlabel[i]] += 1
                    break
            if error == 'span':
                wrongSpan.append(entity)
            elif error == 'margin':
                wrongmargin.append(entity)
            else:
                if 'other' not in wrongmap[type]:
                    wrongmap[type]['other'] = 1
                else:
                    wrongmap[type]['other'] += 1
                allSpanerror.append(entity)
    for index,entity in enumerate(gold):
        if entity not in pred:
            type = entity['type']
            error = 'span'
            for i in entity['offset']:
                if predlabel[i] != 0:
                    error = 'margin'
                    break
            if error == 'span':
                newentity = copy.deepcopy(entity)
                unpred.append(newentity)
    return wrongSpan,wrongmargin,unpred,allSpanerror,wrongmap

def macroupdate(f1,wrongmap,errorinfo,fullrecall,macrof1=None,macrowrongmap=None,macroerrorinfo=None):
    if macrof1 is None:
        macrof1 = {
            'p': [f1['p']],
            'r': [f1['r']],
            'f1': [f1['f1']],
            'recall': [fullrecall]
        }
        if 'typef1' in f1:
            macrof1['typef1'] = {}
            for label in f1['typef1']:
                macrof1['typef1'][label] = {}
                macrof1['typef1'][label]['p'] = [f1['typef1'][label]['p']]
                macrof1['typef1'][label]['r'] = [f1['typef1'][label]['r']]
                macrof1['typef1'][label]['f1'] = [f1['typef1'][label]['f1']]
        macrowrongmap = wrongmap
        macroerrorinfo = errorinfo
    else:
        macrof1['p'].append(f1['p'])
        macrof1['r'].append(f1['r'])
        macrof1['f1'].append(f1['f1'])
        macrof1['recall'].append(fullrecall)
        if 'typef1' in macrof1:
            for label in f1['typef1']:
                if label not in macrof1['typef1']:
                    macrof1['typef1'][label] = {}
                    macrof1['typef1'][label]['p'] = []
                    macrof1['typef1'][label]['r'] = []
                    macrof1['typef1'][label]['f1'] = []
                macrof1['typef1'][label]['p'].append(f1['typef1'][label]['p'])
                macrof1['typef1'][label]['r'].append(f1['typef1'][label]['r'])
                macrof1['typef1'][label]['f1'].append(f1['typef1'][label]['f1'])
        for element in macrowrongmap:
            if element in wrongmap:
                for label in wrongmap[element]:
                    if label not in macrowrongmap[element]:
                        macrowrongmap[element][label] = wrongmap[element][label]
                    else:
                        macrowrongmap[element][label] += wrongmap[element][label]
        for element in macroerrorinfo:
            macroerrorinfo[element] += errorinfo[element]
    return macrof1,macrowrongmap,macroerrorinfo
    


def getmetric(prednum,goldnum,tt,classprednum=None,classgoldnum=None,classtt=None):
    p = 0
    r = 0
    f1 = 0
    if prednum > 0:
        p = tt/prednum
    if goldnum > 0:
        r = tt/goldnum
    if p > 0 and r > 0:
        f1 = 2*p * r / (p+r)
    result = {'p':p,'r':r,'f1':f1}
    if classprednum is not None:
        result['typef1'] = {}
        for label in classprednum:
            p = 0
            r = 0
            f1 = 0
            if classprednum[label] > 0:
                p = classtt[label]/classprednum[label]
            if classgoldnum[label] > 0:
                r = classtt[label]/classgoldnum[label]
            if p > 0 and r > 0:
                f1 = 2*p * r / (p+r)
            result['typef1'][label] = {'p':p,'r':r,'f1':f1}
    return result

def filterpred(preds):
    if len(preds) == 1:
        return preds[0]
    return getstat(preds)

def getstat(preds):
    stats = []
    instance = []
    for pred in preds:
        for entity in pred:
            if entity not in instance:
                instance.append(entity)
                stats.append(1)
            else:
                index = instance.index(entity)
                stats[index] += 1
    scores = np.array(stats)
    indexs = np.argsort(-scores)
    used_offset = []
    newpreds = []
    for si,i in enumerate(indexs):
        if 'offset' in instance[i]:
            flag = True
            for index in instance[i]['offset']:
                if index in used_offset:
                    flag = False
                    break
            if flag:
                newpreds.append(instance[i])
                for index in instance[i]['offset']:
                    used_offset.append(index)
        else:
            newpreds.append(instance[i])
    return newpreds

def evaluefunc(resultfile):

    f1 = []

    lastpred = []
    lastgold = []
    lastdata = 0
    lastindex = 0
    prednum = 0
    goldnum = 0
    tt = 0

    num_wrongspan = 0
    num_wrongmargin = 0
    num_unpred = 0
    num_allspanerror = 0
    num_wronglabel = 0

    errors = []

    classprednum = {}
    classgoldnum = {}
    classtt = {}

    
    error = {
        'wronglabel':[],
    }
    results = []
    targetlabel = {}
    with open(resultfile) as f:
        for line in f:
            line = json.loads(line)
            for label in line['targetlabels']:
                if label not in targetlabel:
                    targetlabel[label] = line['targetlabels'][label]
            results.append(line)
    key = list(targetlabel.keys())[0]
    if targetlabel[key] != 'spot' and targetlabel[key] != 'asoc':
        targetlabel = [targetlabel[i] for i in targetlabel]
    wrongmap = {}
    for label in targetlabel:
        classprednum[label] = 0
        classgoldnum[label] = 0
        classtt[label] = 0
        wrongmap[label] = {}            
    results.append({'index':-1,'pred':[],'gold':[]})

    fullrecall = 0
    fullgold = 0

    for result in results:
        index = result['index']
        pred = result['pred']
        gold = result['gold']
        unipreds = []
        
        for i in range(len(pred)):
            entity = pred[i]
            if entity not in unipreds:
                lowtargets = {i.lower():i for i in result['targetlabels']}
                if entity['type'] not in result['targetlabels'] and entity['type'] in lowtargets:
                    entity['type'] = lowtargets[entity['type']]
                unipreds.append(entity)
        unigolds = []
        for entity in gold:
            if entity not in unigolds:
                unigolds.append(entity)
        result['pred'] = unipreds
        result['gold'] = unigolds
        if index == lastindex:
            # for same instance, append pred and gold
            lastdata = result

            pred = result['pred']
            gold = result['gold']
            lastpred.append(pred)
            lastgold.append(gold)
            continue
        else:
            # for new instance, evalue last instance
            data = lastdata
            gold = lastgold

            error = {'wronglabel':[]}
            error['index'] = lastindex

            unipreds = []
            for eachpred in lastpred:
                for entity in eachpred:
                    if entity not in unipreds:
                        unipreds.append(entity)
            unigolds = []
            for eachgold in lastgold:
                for entity in eachgold:
                    if entity not in unigolds:
                        unigolds.append(entity)
            fullgold += len(unigolds)
            for entity in unipreds:
                if entity in unigolds:
                    fullrecall += 1
            lastpred = filterpred(lastpred)
            gold = unigolds

            error['gold'] = gold
            error['pred'] = lastpred
            prednum += len(lastpred)
            goldnum += len(gold)
            for entity in gold:
                classgoldnum[entity['type']] += 1
            for entity in lastpred:
                classprednum[entity['type']] += 1
            
            predentitytext = [[j['text'],j['offset']] if 'text' in j else None for j in lastpred]

            for entityindex,entity in enumerate(gold):
                if entity in lastpred:
                    tt += 1
                    classtt[entity['type']] += 1
                else:
                    if 'text' in entity and [entity['text'],entity['offset']] in predentitytext:
                        newentity = copy.deepcopy(entity)
                        error['wronglabel'].append(newentity)
                        num_wronglabel += 1
            wrongspan,wrongmargin,unpred,allspanerror,wrongmap = geterror(lastpred,gold,900,wrongmap)
            
            error['wrongspan'] = wrongspan
            error['wrongmargin'] = wrongmargin
            error['unpred'] = unpred
            error['allspanerror'] = allspanerror
            errors.append(error)
            num_wrongspan += len(wrongspan)
            num_wrongmargin += len(wrongmargin)
            num_unpred += len(unpred)
            num_allspanerror += len(allspanerror)

            pred = result['pred']
            lastpred = [pred]
            lastgold = [result['gold']]
            lastindex = index
            lastdata = result
    f1 = getmetric(prednum,goldnum,tt,classprednum,classgoldnum,classtt)
    errorinfo = {
        'wronglabel':num_wronglabel,
        'wrongspan':num_wrongspan,
        'wrongmargin':num_wrongmargin,
        'unpred':num_unpred,
        'allspanerror':num_allspanerror,
    }
    return errors,f1,wrongmap,errorinfo,fullrecall/fullgold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-modelpath", "--modelpath", type=str)

    parser.add_argument("-resultfile", "--resultfile", type=str, default='prediction')

    parser.add_argument("-outputfile", "--outputfile", type=str, default='')

    args = parser.parse_args()

    macrof1 = None
    macrowrongmap = None
    macroerrorinfo =None
    seeds = os.listdir(args.modelpath)
    for seed in seeds:
        resultpath = args.modelpath + '/' + seed
        resultfile = args.modelpath + '/' + seed + '/' + args.resultfile + '.json'
        if not os.path.exists(resultfile):
            continue
        try:
            errors,f1,wrongmap,errorinfo,fullrecall = evaluefunc(resultfile)
            writeresult(resultpath,args.outputfile,f1,wrongmap,errorinfo,errors)
            macrof1,macrowrongmap,macroerrorinfo = macroupdate(f1,wrongmap,errorinfo,fullrecall,macrof1,macrowrongmap,macroerrorinfo)
        except:
            continue
    macroresult = {
        'p':np.mean(macrof1['p']),
        'r':np.mean(macrof1['r']),
        'f1':np.mean(macrof1['f1']),
        'recall':np.mean(macrof1['recall']),
        'std':np.std(macrof1['f1'])
    }
    nums_run = len(macrof1['f1'])
    if 'typef1' in macrof1:
        macroresult['typef1'] = {}
        for label in macrof1['typef1']:
            macroresult['typef1'][label] = {}
            macroresult['typef1'][label]['p'] = np.mean(macrof1['typef1'][label]['p'])
            macroresult['typef1'][label]['r'] = np.mean(macrof1['typef1'][label]['r'])
            macroresult['typef1'][label]['f1'] = np.mean(macrof1['typef1'][label]['f1'])
            macroresult['typef1'][label]['std'] = np.std(macrof1['typef1'][label]['f1'])
    macroresult['origin'] = macrof1
    newmacrowrongmap = {}
    for label in macrowrongmap:
        newmacrowrongmap[label] = sorted(macrowrongmap[label].items(), key=lambda k:k[1], reverse=True)
    writeresult(args.modelpath,args.outputfile,macroresult,newmacrowrongmap,macroerrorinfo)
    sys.stdout.write('run_num: {0}, p:{1:.4f}, r:{2:.4f}, f1: {3:.4f}, std: {4:.4f}'.format(nums_run,macroresult['p'],macroresult['r'],macroresult['f1'],macroresult['std']) + '\r')
    sys.stdout.write('\n')
    print('\n')