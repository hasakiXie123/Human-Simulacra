import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from ..base_api import BaseAPIModel

import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
llmp_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_directory))))
sys.path.append(llmp_directory)

from Config.config import *
from prompt_templates import *
from multi_agent_cognitive_mechanism import Top_agent

PromptType = Union[PromptList, str]
API_BASE =  BASE_URL + '/chat/completions'
 

@MODELS.register_module()
class LLMP_OpenAI(BaseAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        path (str): The name of OpenAI's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        org (str or List[str], optional): OpenAI organization(s). If not
            specified, OpenAI uses the default organization bound to each API
            key. If specified, the orgs will be posted with each request in
            round-robin manner. Defaults to None.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        openai_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.openai.com/v1/chat/completions'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = Model_for_evaluation,
                 max_seq_len: int = Context_length,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 50,
                 key: Union[str, List[str]] = API_KEY,
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = API_BASE,
                 mode: str = 'none',
                 temperature: Optional[float] = None,
                 method: str = 'base_prompt',
                 character: str = "Mary Jones"):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        
        assert method in Method_list
        self.method = method
        
        assert character in Character_list
        self.character_info = {}
        if method == 'base_prompt':
            self.character_info = self.get_info_for_base_prompt(character_name=character)
        elif method == "base_rag":
            self.set_info_for_base_rag(character_name=character)
        elif method == "cognitive":
            self.set_info_for_cognitive(character_name=character)


        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key

        # record invalid keys and skip them when requesting API
        # - keys have insufficient_quota
        self.invalid_keys = set()

        self.key_ctr = 0
        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0
        self.url = openai_api_base
        self.path = path

    def set_info_for_base_rag(self, character_name):
        
        with open(Introductions_Path, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
        for introduction in introductions:
            if introduction["Name"] == character_name:
                self.character_info = introduction
                
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        
        loader = PyPDFLoader(os.path.join(Story_Directory, character_name, character_name + ".pdf"))
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
        )
        docs = text_splitter.split_documents(pages)
        embed_model = OpenAIEmbeddings(
            openai_api_base = OPENAI_BASE_URL, openai_api_key = OPENAI_API_KEY
        )
        
        self.vectorstore = Chroma.from_documents(
            documents=docs, embedding=embed_model, collection_name="openai_embed")
        
    def set_info_for_cognitive(self, character_name):
        with open(Introductions_Path, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
        for introduction in introductions:
            if introduction["Name"] == character_name:
                self.character_info = introduction
        self.top_agent = Top_agent(character_name)
        
    def get_info_for_base_prompt(self, character_name):
        with open(Introductions_Path, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
        for introduction in introductions:
            if introduction["Name"] == character_name:
                return introduction
    
    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic. Defaults to 0.7.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            temperature (float): What sampling temperature to use,
                between 0 and 2. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more
                focused and deterministic.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        # max num token for gpt-3.5-turbo is 4097
        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 128000

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 800 - max_out_len)

        messages = []
        if self.method != 'none':
            sys_prompt = Naive_simulacra_prompt_template.format(
                character_name = self.character_info['Name'],
                basic_information = self.character_info['Basic_infos'],
                personality_traits = self.character_info["Personality_traits"],
                introduction = self.character_info['Content'],
            )
            messages.append({'role': 'system', 'content': sys_prompt})
        # elif self.method == 'base_rag':
        #     sys_prompt = Naive_simulacra_prompt_template.format(
        #         character_name = self.character_info['Name'],
        #         basic_information = self.character_info['Basic_infos'],
        #         introduction = self.character_info['Content'],
        #     )
        #     messages.append({'role': 'system', 'content': sys_prompt})

        if isinstance(input, str):
                messages.append({'role': 'user', 'content': input})    
        else:
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                    
                    if self.method == 'base_rag':
                        matches = re.findall(r'<<<(.*?)>>>', item['prompt'], re.DOTALL)
                        if matches:
                            search_results = self.vectorstore.similarity_search(matches[0], k=3)
                            source_knowledge = "\n".join([x.page_content for x in search_results])
                            sys_prompt = Naive_rag_simulacra_prompt_template.format(
                                source_knowledge = source_knowledge
                            )
                            messages.append({'role': 'system', 'content': sys_prompt})
                        else:
                            pass
                        messages.append(msg)
                    if self.method == "cognitive":
                        messages.append(msg)
                        matches = re.findall(r'<<<(.*?)>>>', item['prompt'], re.DOTALL)
                        if matches:
                            user_prompt = self.top_agent.evaluation_chat(matches[0])
                            messages.append({'role': 'user', 'content': user_prompt})
                        else:
                            pass
                    else:
                        messages.append(msg)
                    
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                    messages.append(msg)
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                    messages.append(msg)
                

        # Hold out 100 tokens due to potential errors in tiktoken calculation
        max_out_len = min(
            max_out_len, context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        # print("---------------------", messages)
        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]

            header = {
                'Authorization': f'Bearer {key}',
                'content-type': 'application/json',
            }

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            try:
                data = dict(
                    model=self.path,
                    messages=messages,
                    max_tokens=max_out_len,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
                raw_response = requests.post(self.url,
                                             headers=header,
                                             data=json.dumps(data))
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            try:
                return response['choices'][0]['message']['content'].strip()
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    self.logger.error('Find error message in response: ',
                                      str(response['error']))
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        if "localhost" in BASE_URL:
            return 500
        else:
            try:
                enc = self.tiktoken.encoding_for_model(self.path)
                return len(enc.encode(prompt))
            except Exception as e:
                self.logger.error('Find error in token calculation: ',
                                      str(e))
                return 500

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt

