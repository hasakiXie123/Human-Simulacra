import json
import re
import random
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from .base import BaseDataset

# Define different prompts for different question types
_single_choice_prompts = {
        "type": "Single_choice",
        "prefix_prompt": ["You are casually chatting with a friend named Alice. You completely trust her and are willing to share everything you know without reservation.\n",
                          "Alice said: 'I have got a question for you: \n<<< ",
                          " >>>\nCould you choose the correct answer from options A, B, C, and D, and respond in the format of 'The answer is...' ?'"],
}

_multiple_choices_prompts = {
        "type": "Multiple_choices",
        "prefix_prompt": ["You are casually chatting with a friend named Alice. You completely trust her and are willing to share everything you know without reservation.\n",
                          "Alice said: 'I have got a question for you: \n<<< ",
                          " >>>\nCould you pick out the correct answers from options A, B, C, D, E, and F, and respond in the format of 'The answer is...' ?'"],
}


_cloze_prompts = {
        "type": "Cloze",
        "prefix_prompt": ["You are casually chatting with a friend named Alice. You completely trust her and are willing to share everything you know without reservation.\n",
                          "Alice said: 'I have got a question for you: \n<<< ",
                          " >>>\nTake a moment to think about this question and respond in the format of 'The answer is...' '"],
}

_open_ended_prompts = {
        "type": "Open-Ended",
        "prefix_prompt": ["You are casually chatting with a friend named Alice. You completely trust her and are willing to share everything you know without reservation.\n",
                          "Alice said: 'I have got a situational judgement test question here for you to try out. The question is: \n<<< ",
                          " >>>\nPlease imagine that you are in the above scenario. Knowing yourself as you do, describe how you would feel and what you would do (not what you should do). The response should follow the format of motive (the reason for taking action) - emotion (internal feelings) - approach (how to take action) - behavior.' "],
}

_jung_cognitive_function_test_prompts = {
        "type": "Jung cognitive function test",
        "prefix_prompt": ["You are casually chatting with a friend named Alice. You completely trust her and are willing to share everything you know without reservation.\n",
                          "Alice said: 'Do you agree this sentence: \n<<< ",
                          " >>>\nTake a moment to think about this and respond Yes or No."],
}

# Function to construct the prompt based on the question type
def prompt_construct(data):
    if data['type'] == "Cloze":
        sys_prompt = _cloze_prompts['prefix_prompt'][0]
        prefix_prompt = _cloze_prompts['prefix_prompt'][1] + data["question"]  + _cloze_prompts['prefix_prompt'][2]
        return sys_prompt, prefix_prompt
    elif data['type'] == "Single_choice":
        sys_prompt = _single_choice_prompts['prefix_prompt'][0]
        prefix_prompt = _single_choice_prompts['prefix_prompt'][1] + data["question"]  + " "
        options = data["options"]
        random.shuffle(options) # Shuffle options to avoid position bias
        prefix_prompt += " ".join(options)
        prefix_prompt += _single_choice_prompts['prefix_prompt'][2]
        return sys_prompt, prefix_prompt
    elif data['type'] == "Multiple_choices":
        sys_prompt = _multiple_choices_prompts['prefix_prompt'][0]
        prefix_prompt = _multiple_choices_prompts['prefix_prompt'][1] + data["question"]  + " "
        # shuffle for position bias
        options = data["options"]
        random.shuffle(options)
        prefix_prompt += " ".join(options)
        prefix_prompt += _multiple_choices_prompts['prefix_prompt'][2]
        return sys_prompt, prefix_prompt
    elif data['type'] == "Jung_cognitive_function_test":
        sys_prompt = _jung_cognitive_function_test_prompts['prefix_prompt'][0]
        prefix_prompt = _jung_cognitive_function_test_prompts['prefix_prompt'][1] + data["question"]  + _jung_cognitive_function_test_prompts['prefix_prompt'][2]
        return sys_prompt, prefix_prompt
    else: ## open-ended
        sys_prompt = _open_ended_prompts['prefix_prompt'][0]
        prefix_prompt = _open_ended_prompts['prefix_prompt'][1] + data["question"]  + _open_ended_prompts['prefix_prompt'][2]
        return sys_prompt, prefix_prompt
        
        

@LOAD_DATASET.register_module()
class LLMPDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        LLMP_data = []
        # path = path.replace(" ", "_")
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data).to_list()
        for data in dataset:
            sys_prompt, prefix_prompt = prompt_construct(data)
            data['sys_prompt'] = sys_prompt
            data['prefix_prompt'] = prefix_prompt
            LLMP_data.append(data)
        dataset = Dataset.from_list(LLMP_data) 
        return dataset


valid_LLMP_question_types = [
    'Cloze', 'Single_choice', 'Multiple_choices', 'Open-Ended', 'Jung_cognitive_function_test' ## Five questions for each type
]

class LLMPEvaluator(BaseEvaluator):

    def __init__(self, question_type) -> None:
        super().__init__()
        assert question_type in valid_LLMP_question_types
        self.question_type = question_type
        self.score4question = 2
        
    # Post-process model output to extract the answer
    def do_predictions_postprocess(self, model_output, answer_lenth=None):
        if self.question_type == 'Single_choice':
            model_answer = []
            index = model_output.find("The answer is")
            if index != -1:
                temp_output = model_output[index: ]
            else:
                temp_output = model_output
            temp = re.findall(r'\b[A-F]\b', temp_output) ## 搜索"A"和 "F"之间最后出现的字符
            if len(temp) != 0:
                model_answer.append(temp[0])

        elif self.question_type == 'Multiple_choices':
            model_answer = []
            index = model_output.find("The answer is")
            if index != -1:
                temp_output = model_output[index: ]
            else:
                temp_output = model_output
            if len(re.findall(r'\b[A-F]\b', temp_output)) > 0:
                    for t in re.findall(r'\b[A-F]\b', temp_output):
                        model_answer.append(t)
            
        elif self.question_type == 'cloze':
            index = model_output.find("The answer is")
            if index != -1:
                model_answer = model_output.lower()[index: ]
            else:
                model_answer = model_output.lower()                
        else: ## Jung_cognitive_function_test
            model_output = model_output.lower()
            if "yes" in model_output:
                model_answer = 1
            else: # no or invalid
                model_answer = 0

        return model_answer

    def ensure_same_length(self, pred, refr):
        if len(pred) == len(refr):
            return pred
        return ['Z'] * len(refr)

    def score(self, predictions, references):
        print("scoring")
        if self.question_type not in [
                'Cloze', 'Single_choice', 'Multiple_choices', ""
        ]:
            return {'score': 0} ## open-ended does not count
        elif self.question_type == 'Cloze':
            correct_score, total_score = 0, 0
            for pred, refr in zip(predictions, references):
                pred = self.do_predictions_postprocess(pred)
                for r in refr:
                    r = r.lower()
                    if r in pred:
                        correct_score += self.score4question
                        print("pred", pred)
                        print("refr", r)
            return {'score': correct_score} 

                
        elif self.question_type == 'Single_choice':
            correct_score, total_score = 0, 0
            for pred, refr in zip(predictions, references):
                pred = self.do_predictions_postprocess(pred)
                pred = self.ensure_same_length(pred, refr)
                for p, r in zip(pred, refr):
                    if p == r:
                        correct_score += self.score4question
            return {'score': correct_score}
            
        else: ## Multiple_choices
            # Points are deducted for wrong choices, with a minimum of zero
            correct_score, total_score = 0, 0
            for pred, refr in zip(predictions, references):
                pred = self.do_predictions_postprocess(pred)

                pred = set(pred) ## remove duplicate values
                for p in pred:
                    if p in refr:
                        correct_score += self.score4question
                    else:
                        correct_score -= self.score4question

                if correct_score < 0 :
                    correct_score = 0
                total_score += correct_score
                correct_score = 0
            
            return {'score': total_score}
        

# Register evaluators for each question type
for question_type in valid_LLMP_question_types:
    # fix classic closure problem
    def _LLMP_register(question_type):
        ICL_EVALUATORS.register_module(
            name='LLMPEvaluator' + '_' + question_type,
            module=lambda *args, **kwargs: LLMPEvaluator(
                question_type=question_type, *args, **kwargs))

    _LLMP_register(question_type)
