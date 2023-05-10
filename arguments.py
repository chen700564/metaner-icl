#!/usr/bin/env python
# -*- coding:utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Arguments:

    plm: str = field(
        metadata={
            "help": "Pretrained model"
        },
    )
    dataset: Optional[str] = field(
        default="pretrain", metadata={"help": "path of pre-training dataset"}
    )
    formatsconfig: Optional[str] = field(
        default="config/formats/metaner.yaml", metadata={"help": "config file for icl formats"}
    )
    randomseed: Optional[int] = field(
        default=2333, metadata={"help": "random seed"}
    )
    context_num: Optional[int] = field(
        default=5, metadata={"help": "number of context instances"}
    )
    input_maxlength: Optional[int] = field(
        default=100, metadata={"help": "max token id length for input instance"}
    )
    max_instance_length: Optional[int] = field(
        default=60, metadata={"help": "max token id length for context instaces"}
    )
    enhance: Optional[str] = field(
        default="anonymization_nerpretrain_lmpretrain_metapretrain", metadata={"help": "anonymization means using type anonymization. nerpretrain means pretrain ner task. lmpretrain means pretrain using lm dataset. metapretrain means using meta function pretraining."}
    )
    debugfile: Optional[str] = field(
        default="None", metadata={"help": "the file for debug"}
    )

@dataclass
class ArgumentsForTest:

    plm: str = field(
        metadata={
            "help": "Pretrained model"
        },
    )
    formatsconfig: Optional[str] = field(
        default="config/formats/metaner.yaml", metadata={"help": "config file for icl"}
    )
    randomseed: Optional[int] = field(
        default=2333, metadata={"help": "random seed"}
    )
    context_num: Optional[int] = field(
        default=5, metadata={"help": "number of context instances"}
    )
    input_maxlength: Optional[int] = field(
        default=100, metadata={"help": "max token id length for input instance"}
    )
    max_instance_length: Optional[int] = field(
        default=60, metadata={"help": "max token id length for context instaces"}
    )
    testset: Optional[str] = field(
        default="data/conll03", metadata={"help": "testset file"}
    )
    predictfile: Optional[str] = field(
        default="prediction", metadata={"help": "name of predict file"}
    )
    shot_num: Optional[int] = field(
        default=5, metadata={"help": "number of instances for each type"}
    )
    debugfile: Optional[str] = field(
        default="None", metadata={"help": "the file for debug"}
    )