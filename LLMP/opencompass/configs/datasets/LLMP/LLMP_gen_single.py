from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

with read_base():
    from .LLMP_gen_0001 import LLMP_datasets, LLMP_models
    
datasets = [*LLMP_datasets]

models = [*LLMP_models]
