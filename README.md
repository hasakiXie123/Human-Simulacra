# Human Simulacra: Benchmarking the Personification of Large Language Models 
<div align="center">
    <a>Qiujie Xie<sup>1</sup></a>&emsp;
    <a>Qiming Feng<sup>*1</sup></a>&emsp;
    <a>Tianqi Zhang<sup>*2</sup></a>&emsp;
    <a>Qingqiu Li<sup>1</sup></a>&emsp; <br>
    <a>Linyi Yang<sup>3</sup></a>&emsp;
    <a>Yuejie Zhang<sup>†1</sup></a>&emsp;
    <a>Rui Feng<sup>†1</sup></a>&emsp; <br>
    <a>Liang He<sup>4</sup></a>&emsp;
    <a>Shang Gao<sup>5</sup></a>&emsp;
    <a>Yue Zhang<sup>3</sup></a>&emsp;
    <p> <sup>1</sup> School of Computer Science, Shanghai Key Laboratory of Intelligent Information Processing, Fudan University, 
    <sup>2</sup> School of Electronic and Information Engineering, Tongji University, 
    <sup>3</sup> School of Engineering, Westlake University,  
    <sup>4</sup> School of Computer Science and Technology, Shanghai Key Laboratory of Mental Health and Psychological Crisis Intervention, East China Normal University,  
    <sup>5</sup> School of Information Technology, Deakin University</p>
</div>

This repository contains the code and data for Human Simulacra. 
- 🌟For data📖, please refer to [The Human Simulacra Dataset](#dataset).
- 🌟For reproduction🔍, please refer to [Reproduction](#reproduction).
- 🌟If you want to make your own characters with a unique life story🪄, please refer to [Character Customization](#make).
- 🌟If you want to edit existing characters🔨, please refer to [Character Modification](#edit).

For any questions or concerns regarding this dataset, please feel free to reach out to us. We appreciate your interest and are eager to assist😊.

![Task](https://img.shields.io/badge/Task-Human%20Simulation-blue.svg)
![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC--SA%204.0-blue.svg)

## Overview
Large language models (LLMs) are recognized as systems that closely mimic aspects of human intelligence. This capability has attracted attention from the social science community, who see the potential in leveraging LLMs to replace human participants in experiments, thereby reducing research costs and complexity. In this paper, **we introduce a benchmark for large language models personification, including a strategy for constructing virtual characters' life stories from the ground up, a Multi-Agent Cognitive Mechanism capable of simulating human cognitive processes, and a psychology-guided evaluation method to assess human simulations from both self and observational perspectives.** Experimental results demonstrate that our constructed simulacra can produce personified responses that align with their target characters. 

<div align="center">
<a href="https://github.com/hasakiXie123/Human-Simulacra/">
    <img src="figures/dataset.jpg" alt="dataset" width="600" class="center">
</a>
  
<a>Figure 1: **Left**: Information model for virtual characters. **Right**: Process of constructing life stories for target characters using semi-automated strategy. At each step, humans are involved in thoroughly reviewing the generated content, ensuring that it is free from biases and harmful information.</a>
</div>

<div align="center">
<a href="https://github.com/hasakiXie123/Human-Simulacra/">
    <img src="figures/model.jpg" alt="model" width="600" class="center">
</a>
  
<a>Figure 2: **Multi-Agent Cognitive Mechanism.** This mechanism involves four LLM-driven agents. The **Thinking Agent** / **Emotion Agent** handles logical/emotional analysis and logical/emotional memory construction. The **Memory Agent** manages retrieval of memories, while the **Top Agent** coordinates all activities. Upon receiving a stimulus, these agents collaborate to generate appropriate responses, simulating complex human cognitive processes.</a>
</div>

<div align="center">
<a href="https://github.com/hasakiXie123/Human-Simulacra/">
    <img src="figures/evaluation.jpg" alt="evaluation" width="600" class="center">
</a>
  
<a>Figure 3: **Psychology-guided evaluation.** The self report assesses the simulacra‘s self-awareness through character-specific questions based on their life stories. The observer report evaluates the realism of simulacra by creating scenario-based assessments analyzed by human judges. </a>
</div>

<a name="dataset"></a>
## The Human Simulacra Dataset📖
Using the semi-automated strategy proposed in the paper, we create a multi-grained virtual character dataset named **Human Simulacra**. It comprises about 129k texts across 11 virtual characters, with each character having their unique attributes, biography, and story. We place the data in LLMP/Characters. 
> [!NOTE]  
> In each story generation, we thoroughly review the generated content to ensure it is free from biases, discrimination, or harmful information. If you find any harmful content, please notify us.

The structure is explained as follows:
```
├─Attributes ## The candidate pools for the attributes of characters.
├─Memories ## The long-term memories of characters.
│  ├─Mary Jones
│  ├─Sara Ochoa
|  └─...
├─Questions ## The evaluation questions for characters.
│  ├─Mary Jones
│  ├─Sara Ochoa
|  └─...
└─Stories ## The life stories of characters.
    ├─Mary Jones
    ├─Sara Ochoa
    └─...
character_introductions.json ## The biographies of 11 characters.
character_profiles.json ## The profiles of 11 characters.
```

<a name="reproduction"></a>
## Reproduction🔍

<a name="reproduction-install"></a>
### Installation
1. To get started, first clone the repository and setup the environment:
```
## Set up the environment
git clone https://github.com/hasakiXie123/Human-Simulacra.git
cd Human-Simulacra

## Install required packages
conda env create -f environment.yml

## Activate the environment
conda activate LLMP
cd LLMP

## Install OpenCompass for Psychology-guided evaluation
cd opencompass
pip install -e .
cd ../
```
2. Adjust the path section of Config/config.py to suit your situation:
```python
## for example, if your current directory is "/root/abc/Human-Simulacra/LLMP"
for path/directory in the path section of Config/config.py:
    replace "/root/Desktop/LLMP" with "/root/abc/Human-Simulacra/LLMP"
    # e.g., Attributes_Directory = "/root/abc/Human-Simulacra/LLMP/Characters/Attributes"
```
3. Adjust the following lines of Config/config.py to suit your situation:
```
Model_for_evaluation = "gpt-4-1106-preview" # choose the base model for simulacra
Model_for_agent = "gpt-4-1106-preview" ## agent in MACM
OPENAI_API_KEY = "Your OPENAI_API_KEY"
OPENAI_BASE_URL = "https://api.openai.com/v1"
API_KEY = "Your API_KEY"
BASE_URL = "https://api.openai.com/v1" # or any api company
```
If you want to use a local model, for example, a Llama-2-7b model based on the [Fastchat library](https://github.com/lm-sys/FastChat):
```
Model_for_evaluation = "Llama-2-7b" # choose the base model for simulacra
Model_for_agent = "Llama-2-7b" ## agent in MACM
OPENAI_API_KEY = "EMPTY"
OPENAI_BASE_URL = "http://localhost:8000/v1" 
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8000/v1" 
```
### Chat with simulacrum based on MACM/Prompt/RAG
Run the following commands:
```
## Chat with simulacrum based on the MACM method. The simulacrum is simulating Mary Jones.
python multi_agent_cognitive_mechanism.py --character_name "Mary Jones" --method macm
## Chat with simulacrum based on the prompt method.
python multi_agent_cognitive_mechanism.py --character_name "Mary Jones" --method prompt
## Chat with simulacrum based on the rag method.
python multi_agent_cognitive_mechanism.py --character_name "Mary Jones" --method rag
```

### Psychology-guided evaluation
In the proposed Psychology-guided evaluation, each character is tested by its own set of questionnaires containing cloze, single-choice and multiple-choice questions. You can find the questions in LLMP/Characters/Questions.

We build the evaluation code based on the [OpenCompass library](https://github.com/open-compass/opencompass). If you want to evaluate simulacra of existing characters:
1. Modify the following lines of LLMP/opencompass/configs/datasets/LLMP/LLMP_gen_0001.py:
```python
Question_Directory = "/root/abc/Human-Simulacra/LLMP/Characters/Questions" ## Please ensure consistency with your config file
## Evaluate blank simulacrum. The simulacrum does not know any information about the target character "Mary Jones".
Character_name = "Mary Jones" 
Method_list = ["none"]
## Evaluate prompt-based simulacrum. The simulacrum is simulating Mary Jones.
Character_name = "Mary Jones" 
Method_list = ["base_prompt"]
## Evaluate rag-based simulacrum. The simulacrum is simulating Mary Jones.
Character_name = "Mary Jones" 
Method_list = ["base_rag"]
## Evaluate MACM-based simulacrum. The simulacrum is simulating Mary Jones.
Character_name = "Mary Jones" 
Method_list = ["cognitive"]
```
2. Run the following commands to start the evaluation. The result will be saved in Outputs/demo:
```
python opencompass/run.py /your current directory/LLMP/opencompass/configs/datasets/LLMP/LLMP_gen_single.py -w /your current directory/LLMP/Outputs/demo
```
If you want to evaluate simulacra of self-made characters, which are constructed by using our semi-automated strategy:
1. Place the life story of the character in LLMP/Characters/Stories/(name)/
2. Prepare questionnaires for the character. We provide templates for each question type at LLMP/Characters/Questions.
3. Modify the following lines of LLMP/opencompass/configs/datasets/LLMP/LLMP_gen_0001.py:
```python
## Evaluate blank simulacrum. The simulacrum does not know any information about the target character.
Character_name = "(name)" 
Method_list = ["none"]
## Evaluate prompt-based simulacrum. The simulacrum is simulating (name).
Character_name = "(name)" 
Method_list = ["base_prompt"]
...
```
4. Adjust the following lines of Config/config.py to append your character to the existing list:
```
Character_list = ["Mary Jones", ..., "Marsh Zhaleh", (name)]
```
5. Run the following commands to start the evaluation. The result will be saved in Outputs/demo:
```
python opencompass/run.py /your current directory/LLMP/opencompass/configs/datasets/LLMP/LLMP_gen_single.py -w /your current directory/LLMP/Outputs/demo
```
### Bandwagon effect replication
We currently support replicating the bandwagon effect with prompt-/RAG-/MACM-based simulacra, run the following commands:
```
python bandwagon_effect.py --character_name "Mary Jones" --method "macm"
## Controlled experiment (without group pressure)
python bandwagon_effect.py --character_name "Mary Jones" --method "macm" --control
```
### An alternative method for human evaluation (Observer report)
To facilitate future research on this benchmark, we further proposed an alternative method for human evaluation. This method involves using the 640 personality descriptions constructed in this paper to perform an eight-dimensional personality test and comparing the results to the assigned personality for similarity. In experiments, the scores calculated using this method had a **Pearson correlation coefficient of 0.810 and an Intraclass Correlation Coefficient of 0.877** with the human evaluation scores.

Please mmodify the following lines of LLMP/opencompass/configs/datasets/LLMP/LLMP_gen_0001.py first:
```python
Question_type = ["Jung_cognitive_function_test"]
Character_name = "Mary Jones" 
Method_list = ["base_prompt"] # or base_rag/cognitive
```

Then, run the following commands:
```
## Please ensure that both if_extract and if_rerun are set to True when running the file for the first time.
python Jung_cognitive_function_test.py --character_name "Mary Jones" --method "prompt" --if_extract --if_rerun 
```

<a name="make"></a>
## Make your own characters🪄
Please complete the steps in [Reproduction-Installation](#reproduction-install) first. Then, download the weight of summarization model (we use [pszemraj/led-large-book-summary](https://huggingface.co/pszemraj/led-large-book-summary) and [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization/tree/main) from Hugging Face) and embedding model ([sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) from Hugging Face), put them into LLMP/LLMs. Finally, modify the following lines of Config/config.py to suit your situation:
```
Model_for_data = "gpt-3.5-turbo" # model for data generation
Context_length = 128000 # the context length for data generation model
Iteration_for_story = 10
Iteration_threshold = 50 # score threshold
Window_size = 5 # if story length > Context_length, context = [-Window_size, Window_size]
Continued = False ## for continued story generation
Continued_story_path = "/root/Desktop/LLMP/Characters/Stories/Erica Walker/Erica Walker-19.txt" ## for continued story generation, if you want use this, please set the argument-Continued to True
```
### Character Profiles
Run the following commands to generate 100 candidate profiles (varied in quality). The result will be saved in Candidate_character_profiles.json, and you can select your favorite character profiles to proceed to the next step.
```
python generate_character_profile.py
```
### Character Biography
Run the following commands to generate a short biography for each candidate character within Candidate_character_profiles.json. The result will be saved in Candidate_character_introductions.json.
```
python generate_character_introduction.py
```

<a name="#life-story-generation"></a>
### Life Story Generation
Run the following commands to generate the life stories. You can choose to generate life stories for all candidate characters at once, or specify a particular character to be generated. 
```
## generate all at once
python generate_character_life_story.py
## generate the life story of Mary Jones
python generate_character_life_story.py --character_name "Mary Jones"
```
To ensure the quality of the life story, we set Iteration_for_story to 10, which means that after every ten rounds of Iteration, we will suspend the generation and manually review the generated life story to make sure that it does not deviate from the character's personality and does not contain harmful content. For your reference, each life story in the Human Simulacra dataset is expanded through at least **50 rounds** of iteration and incurs at least **three days** of human effort for content reviewing.

<a name="#continued-generation"></a>
### Continued Generation
Modify the following lines of Config/config.py to suit your situation:
```
Continued = True ## for continued story generation
Continued_story_path = "/root/Desktop/LLMP/Characters/Stories/Erica Walker/Mary Jones-19.txt" ## for continued story generation, if you want use this, please set the argument-Continued to True
```
Run the following commands to continue iterating the story.
```
## generate the life story of Mary Jones
python generate_character_life_story.py --character_name "Mary Jones"
```

<a name="edit"></a>
## Edit existing characters🔨
We now support adding new character attributes (e.g., favorite color, favorite author) or new life experiences to an existing character😊. 

Before you proceed, please first confirm whether the character you wish to modify originates from the Human Simulacra dataset or if it is among the self-made characters, which are constructed using our semi-automated strategy. If it's the latter, please complete the following steps:

1. Place the life story of the character in LLMP/Characters/Stories/(name)/
2. Append the introduction of the character to LLMP/Characters/character_introductions.json

Then, run the following commands to add new character attributes or new life experiences.
```
agent = Top_agent("Mary Jones")
agent.add_new_attributes({"Favorite Color": "Green", "Favorite Author": "Ada Lovelace"})
agent.add_long_memory("Mary's favorite color is green, and her favorite author is Ada Lovelace.")
```

Now😊, you can ask Mary what her favorite color is, and she will respond like this: ''It's green. Always loved how it reminds me of nature and the outdoors.'' 

We further support iterating the life story of a character based on newly added character attributes. You can proceed with the iteration following the tutorial in [Continued-Generation](#continued-generation). However, if you wish for the newly added character attributes to influence the entire life story of the character, we recommend regenerating the story from the beginning ([Life-Story-Generation](#life-story-generation)) to ensure the quality of the narrative.

## Maintenance Log
| Dates      | Maintenance Content         |
|------------|-----------------|
| Jun 8, 2024 | 	Update code, fix bugs.   |
| Jun 11, 2024 |    Release new features, support replicating the bandwagon effect with prompt-/RAG-/MACM-based simulacra.         |
| Jun 12, 2024 |    Release new features, support adding new character attributes or new life experiences to an existing character.         |
| Jun 15, 2024 |    Release new features, support performing an eight-dimensional personality test to the character.         |


## Contributing
Contributions to enhance the usability and quality of this dataset are always welcomed. If you're interested in contributing, feel free to fork this repository, make your changes, and then submit a pull request. For significant changes, please first open an issue to discuss the proposed alterations.

## Acknowledgements
- 🌟 We appreciate the guidance of the psychologists from Shanghai Key Laboratory of Mental Health and Psychological Crisis Intervention, School of Computer Science and Technology, East China Normal University throughout the process.
- 🌟 We build the evaluation code based on the [OpenCompass library](https://github.com/open-compass/opencompass). We appreciate their well-built code, which helped us save some time.
## Citation
If you find our work useful, please cite our [paper](https://arxiv.org/abs/2402.18180)😊:
```bibtex
@article{xie2024human,
  title={Human Simulacra: Benchmarking the Personification of Large Language Models}, 
  author={Xie, Qiuejie and Feng, Qiming and Zhang, Tianqi and Li, Qingqiu and Yang, Linyi and Zhang, Yuejie and Feng, Rui and He, Liang and Gao, Shang and Zhang, Yue},
  journal={arXiv preprint arXiv:2402.18180},
  year={2024}
}
```
