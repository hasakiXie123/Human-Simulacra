from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LLMPDataset
from opencompass.models.LLMP import LLMP_OpenAI



Question_Directory = "/root/Desktop/LLMP/Characters/Questions"
Character_name = "Mary Jones" 

Question_type = [
    "Cloze",
    "Single_choice",
    "Multiple_choices",
    "Open-Ended"
]

Method_list = [
    "base_prompt",
    # "none",
    # "base_rag",
    # "cognitive"
]

LLMP_datasets = []
LLMP_models = []

 
for q_type in Question_type:
    _reader_cfg = {
            "input_columns": ['question', 'options', 'sys_prompt', 'prefix_prompt'],
            "output_column": 'answer',
    }
    _infer_cfg = {
            "ice_template": {
                "type": PromptTemplate,
                "template": {
                    "round": [
                        {
                        "role": "SYSTEM",
                        "prompt": '{sys_prompt}'
                        },
                        {
                        "role": "HUMAN",
                        "prompt": '{prefix_prompt}'
                        },
                    ]
                },
                "ice_token": "</E>"
            },
            "retriever": {
                "type": ZeroRetriever
            },
            "inferencer": {
                "type": GenInferencer,
                "max_out_len": 1024,
            }
    }
    _eval_cfg = {
            "evaluator": {
                "type": "LLMPEvaluator" + "_" + q_type,
            },
            "pred_role": "BOT",
    }
    _dataset = {
            "type": LLMPDataset,
            "abbr": "LLMP_" + Character_name + "_" + q_type,
            "path": Question_Directory + "/" + Character_name + "/" + q_type + ".json",
            "reader_cfg": _reader_cfg,
            "infer_cfg": _infer_cfg,
            "eval_cfg": _eval_cfg,
    }
    LLMP_datasets.append(_dataset) 

for method in Method_list:    
    meta_template=dict(
        round=[
            dict(role='SYSTEM', api_role='SYSTEM'),
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
    )
    
    _model = {
            "abbr" : Character_name,
            "type" : LLMP_OpenAI, 
            "meta_template" : meta_template,
            "query_per_second" : 1,
            "max_out_len" : 2048, 
            "max_seq_len" : 128000, 
            "batch_size" : 8,
            "method" : method,
            "character" : Character_name
    }
    LLMP_models.append(_model)

_temporary_variables = [k for k in globals() if k.startswith('_')]
for _t in _temporary_variables:
    del globals()[_t]
del _temporary_variables, _t
