import copy
import random

    
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

def getnolabel(targetlabels):
    mapping = {}
    labelindex = 0
    labelnames = ['<type' + str(i) + '>' for i in range(1, 99)]
    random.shuffle(labelnames)
    for label in targetlabels:
        if label not in mapping:
            flag = True
            if random.random() < 0.2:
                flag = False
            if labelindex >= len(labelnames):
                flag = False
            if flag:
                mapping[label] = labelnames[labelindex]
                labelindex += 1
            else:
                mapping[label] = label
    return mapping
    
def getnerinstance(tokenizer, formats, instance, labels, endid, lm = False, code2name=None):
    instance = getdata(tokenizer, instance,200)
    text = ' '.join(instance['tokens'])
    inputids = tokenizer.encode(text, add_special_tokens = False) + [endid]
    
    positivelabels = []
    for entity in instance['entity']:
        if type(entity['type']) is list:
            for etype in entity['type']:
                if etype not in positivelabels:
                    positivelabels.append(etype)
        else:
            if entity['type'] not in positivelabels:
                positivelabels.append(entity['type'])
    candites = random.sample(labels,min(len(labels),20))
    negativelabels = [i for i in candites if i not in instance['entity']]

    targetlabelnum = random.choice(range(2,15))
    posrate = random.choice(list(range(0,110,10)))
    posnum = int(targetlabelnum * posrate * 0.01)

    labels = random.sample(positivelabels,min(len(positivelabels),posnum))

    if targetlabelnum - len(labels) > 0:
        neglabels = random.sample(negativelabels,min(len(negativelabels),targetlabelnum - len(labels)))
    else:
        neglabels = []
    targetlabels = labels + neglabels

    if lm:
        instance = getlmdata(instance,targetlabels)

    random.shuffle(targetlabels)
    mapping = {i:i for i in targetlabels}
    if code2name is not None and not lm:
        for code in mapping:
            if mapping[code] in code2name:
                mapping[code] = code2name[mapping[code]]

    instruction = getinstruction(targetlabels, mapping)
    instructionid = tokenizer.encode(instruction, add_special_tokens=False)
    outputs,labels = getoutputids(tokenizer, formats, instance,targetlabels,mapping)

    inputids = instructionid + inputids

    instance = {
        'input_ids': inputids,
        'attention_mask': [1] * len(inputids),
        'decoder_input_ids': outputs[:500],
        'labels': labels[:500]
    }
    return instance

def getdata(tokenizer, instance, max_length):
    instance = copy.deepcopy(instance)
    try:
        tokenized_inputs = tokenizer(
            instance['tokens'], max_length=max_length, truncation=True, is_split_into_words=True, add_special_tokens=False
        )
    except:
        tokenized_inputs = tokenizer(
            instance['tokens'], max_length=max_length, truncation=True, add_special_tokens=False
        )
    instance['input_ids'] = tokenized_inputs['input_ids']
    newentities = []
    entities = instance['entity']
    try:
        word_ids = tokenized_inputs.word_ids()
        maxtokennum = max(word_ids)
    except:
        word_ids = None
        maxtokennum = len(instance['tokens'])
    instance['tokens'] = instance['tokens'][:maxtokennum + 1]
    for entity in entities:
        if entity['offset'][-1] <= maxtokennum:
            newentities.append(entity)
    instance['entity'] = newentities
    return instance
    
def getoutputids(tokenizer, formats,context,targetlabels,mapping):
    
    outputids = []
    labels = []
    index = 0
    for entity in context['entity']:
        types = entity['type']
        labelname = ''
        if type(types) is list:
            labelname = [mapping[i] for i in types if i in targetlabels]
            if len(labelname) > 0:
                labelname = ', '.join(labelname)
        else:
            if types in targetlabels:
                labelname = mapping[types]
            
        if len(labelname) > 0:
            text = entity['text'] + ' is ' + labelname + '.'
            if index == 0:
                outputid = tokenizer.encode(text, add_special_tokens=False)
            else:
                outputid = tokenizer.encode(' '+text, add_special_tokens=False)
            labelid = outputid
            outputids += outputid
            labels += labelid
            index += 1
    outputids = formats['entity']['prefix'] + outputids + formats['entity']['end']
    labels = formats['entity']['prefix'] + labels + formats['entity']['end']
    return outputids,labels
    
def sample_contexts(instance_nums,contexts,labels,banindex=-1):
    sampled_contexts = []
    for label in labels:
        indexs = copy.deepcopy(contexts[label])
        selected_indexs = random.sample(indexs,min(instance_nums+1,len(indexs)))
        if banindex > 0:
            selected_indexs = [i for i in selected_indexs if i != banindex]
        sampled_contexts += selected_indexs
    sampled_contexts = random.sample(sampled_contexts, min(instance_nums,len(contexts)))
    sampled_contexts = copy.deepcopy(sampled_contexts)
    return sampled_contexts
    
def getlmdata(instance, targetlabels):
    instance = copy.deepcopy(instance)
    index = 0
    mapping = {}
    labelindex = 0
    labelnames = ['<mask' + str(i) + '>' for i in range(1, 99)]
    random.shuffle(labelnames)
    unmaskedindex = list(range(len(instance['tokens'])))

    for entity in instance['entity']:
        if entity['type'] in targetlabels:
            entity['text'] = labelnames[labelindex]
            instance['tokens'][entity['offset'][0]] = labelnames[labelindex]
            unmaskedindex.remove(entity['offset'][0])
            labelindex += 1
            if labelindex >= len(labelnames):
                break
    if len(unmaskedindex) < 0.85 * len(instance['tokens']) and labelindex < len(labelnames):
        masknum = int(0.85 * len(instance['tokens'])) - len(unmaskedindex)
        maskedindex = random.sample(unmaskedindex,min(len(unmaskedindex),masknum))
        for index in maskedindex:
            instance['tokens'][index] = labelnames[labelindex]
            labelindex += 1
            if labelindex >= len(labelnames):
                break
    return instance

def getinstance(queryinstace, contexts, tokenizer, max_instance_length, formats, targetlabels, enhance, lm= False, code2name= None):

    if 'anonymization' in enhance:
        mapping = getnolabel(targetlabels)
    else:
        mapping = {k:k for k in targetlabels}
    
    if code2name is not None and not lm:
        for code in mapping:
            if mapping[code] in code2name:
                mapping[code] = code2name[mapping[code]]
    
    instruction = getinstruction(targetlabels, mapping)
    instructionid = tokenizer.encode(instruction, add_special_tokens=False)
    instructionindex = list(range(len(instructionid)))

    text = getpredformat(formats['entity']['inputformat'],' '.join(queryinstace['tokens']))
    queryinputids = tokenizer.encode(text, add_special_tokens=False)
    queryindexs = list(range(len(queryinputids)))
    

    iclinputids = []
    icldecodeinputids = []
    fullinputids = instructionid

    newtargetlabels = []
    for context in contexts:
        context = getdata(tokenizer, context, 100)
        if lm:
            context = getlmdata(context,targetlabels)

        text = getpredformat(formats['entity']['inputformat'],' '.join(context['tokens']))
        textids = tokenizer.encode(text, add_special_tokens=False)
        outputids,_ = getoutputids(tokenizer, formats, context,targetlabels,mapping)
        
        if len(fullinputids + textids + outputids + queryinputids) >= max_instance_length:
            continue
        iclinputids.append(textids)
        icldecodeinputids.append(outputids)
        fullinputids = fullinputids + textids + outputids
        for entity in context['entity']:
            if type(entity['type']) is list:
                for etype in entity['type']:
                    if etype in targetlabels:
                        newtargetlabels.append(etype)
            else:
                if entity['type'] in targetlabels:
                    newtargetlabels.append(entity['type'])
    
    indexs = instructionindex + [i + len(fullinputids) for i in queryindexs]
    fullinputids = fullinputids + queryinputids
    queryoutputids,labels = getoutputids(tokenizer, formats, queryinstace,newtargetlabels,mapping)
    
    newinstance = {
        'input_ids': fullinputids,
        'attention_mask': [1] * len(fullinputids),
        'decoder_input_ids': queryoutputids[:500],
        'labels': labels[:500],
        'icl_input_ids': [instructionid + i for i in iclinputids],
        'icl_attention_mask': [[1] * (len(i) + len(instructionid)) for i in iclinputids],
        'icl_decoder_input_ids': icldecodeinputids,
        'query_input_ids': instructionid + queryinputids,
        'query_attention_mask': [1] * len(queryinputids),
        'indexs': indexs,
    }
    if 'metapretrain' not in enhance or lm:
        newinstance['indexs'] = []
    newinstance['tokens'] = queryinstace['tokens']
    newinstance['entity'] = getresult(queryinstace['entity'], newtargetlabels)
    newinstance['mapping'] = mapping
    newinstance['task'] = 'entity'
    newinstance['targetlabel'] = {mapping[i]:'spot' for i in newtargetlabels}
    return newinstance

def getresult(entities,targetlabels):
    newentties = []
    for entity in entities:
        if type(entity['type']) is list:
            for etype in entity['type']:
                if etype in targetlabels:
                    newentties.append({"type":etype,'offset':entity['offset'],'text':entity['text']})
        else:
            newentties.append(entity)
    return newentties
