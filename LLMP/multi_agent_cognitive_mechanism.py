import os
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import openai
import json
import re
import sys

from Config.config import *
from prompt_templates import *

class Memory_agent:
    # The Memory agent is responsible for the following functions:
    # (1) Add long-term memory, stored in Long_memory.json;
    # (2) Add short-term memory, stored in Short_memory.txt;
    # (3) Retrieval: Retrieve the most relevant memory summaries from Index.json using LLM based on the query, then find the corresponding memories in Long_memory.json.
    def __init__(self, character_name, temperature=0.0, api_base=BASE_URL, api_key=API_KEY):
        
        self.name = character_name
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.path = os.path.join(Memory_Directory, character_name)
        
        self.sum = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=Model_for_agent,
            temperature = self.temperature
        )
        self.retrieval = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=Model_for_agent,
            temperature = self.temperature
        )
        self.system_prompt = Memory_agent_system_prompt_template.format(
            character_name = self.name
        )

    def Summary(self, memory_chunk, emotion):
        # Generate a summary of a memory chunk for subsequent memory retrieval.
        system_prompt = Memory_summary_system_prompt_template
        user_prompt = Memory_summary_user_prompt_template.format(
            character_name = self.name,
            memory_chunk = memory_chunk,
            emotion = emotion
        )
        messages = []
        messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        # Generate summary
        summary = self.sum(messages).content
        return summary

    def Index_construction(self, index):
        # Construct the index file (index.json) with the structure: {"num1": "Memory_Summary1", "num2": "Memory_Summary2"....}
        file_path = os.path.join(self.path, "index.json")
        if not os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, 'w') as file:
                json.dump(index, file)  # 写入内容

            print(f"File {file_path} created successfully!")
        else:
            print(f"File {file_path} already exists")
            with open(file_path, 'w') as file:
                json.dump(index, file)

    def Add_long_memory(self, memory):
        # Add long-term memory to long_memory.json
        file_path = os.path.join(self.path, "long_memory.json")  # 连接文件夹路径和文件名
        if not os.path.exists(self.path):  # 检查文件是否存在
            with open(file_path, 'w') as file:
                json.dump(memory, file)  # 写入内容
                
            print(f"File {file_path} created successfully!")
        else:
            print(f"File {file_path} already exists, replace it with a new one!")
            with open(file_path, 'w') as file:
                json.dump(memory, file)

    def Add_short_memory(self, memory):
        # Add short-term memory to short_memory.txt
        file_path = os.path.join(self.path, "short_memory.txt")  # 连接文件夹路径和文件名
        if not os.path.exists(self.path):  # 检查文件是否存在
            with open(file_path, 'w') as file:
                file.write(memory)  # 写入内容
                file.write('\n')
            print(f"File {file_path} created successfully!")
        else:
            print(f"File {file_path} already exists")
            with open(file_path, 'a') as file:
                file.write(memory)
                file.write('\n')

    def Search(self, result_list):
        # Retrieve relevant memories from long_memory.json based on result_list
        file_path = os.path.join(self.path, "long_memory.json")
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        Search_resurt = {}
        Memory_list = []
        key_list = result_list
        for key in key_list:
        # 检查键是否存在于JSON数据中
            if key in json_data:
                # 获取对应键的"Memeory"内容
                memory = json_data[key]["Memory"]
                # 添加到结果列表中
                Memory_list.append(memory)
        num = 0
        limit = 1 ## one most related memory
        for memory in Memory_list:
            num += 1
            if num > limit:
                break
            string_num = str(num).zfill(3)
            Search_resurt[string_num] = memory
        return Search_resurt
    

    def Memory_Retrieval(self, Query):
        # Perform memory retrieval based on LLM, memories are stored in Long_memory.json
        # Memory structure: {"num1": {"Memory_Summary": "xxx", "Memory": {"Memory Content": "xxx", "Thinking": "xxx", "Emotion": "xxx"}}, "num2": {"Memory_Summary": "xxx", "Memory": {"Memory Content": "xxx", "Thinking": "xxx", "Emotion": "xxx"}}, ... }
        # The retrieval process involves LLM determining the most relevant Memory_Summaries for the query, then aggregating the corresponding Memories as the retrieval result.
        messages1 = []
        messages1.append(SystemMessage(content=self.system_prompt))
        messages2 = []
        messages2.append(SystemMessage(content=self.system_prompt))
        index_file_path = os.path.join(self.path, "index.json")
        # Index file name is Index.json, structure: {"num1": "Memory_Summary1", "num2": "Memory_Summary2"....}
        # The retrieval result returns memory segment numbers (num)
        with open(index_file_path, 'r') as file:
            index = json.load(file)
        
        keys = list(index.keys())

        # Split index file into two parts due to potential context length limit, calculate split point
        split_point = len(keys) // 2

       # Use slicing to split the key list
        keys1 = keys[:split_point]
        keys2 = keys[split_point:]

        index1 = {key: index[key] for key in keys1}
        index2 = {key: index[key] for key in keys2}

        user_prompt = Memory_agent_user_prompt_template.format(
            index = index1,
            query = Query
        )
        messages1.append(HumanMessage(content = user_prompt))
        
        user_prompt = Memory_agent_user_prompt_template.format(
            index = index2,
            query = Query
        )
        messages2.append(HumanMessage(content=user_prompt))

        ans = self.retrieval(messages1).content
        ans += self.retrieval(messages2).content
        pattern = r'"\d{3}"'  
        matches = re.findall(pattern, ans)  
        result_list = list(set([match.strip('"') for match in matches]))
        # Find corresponding memories based on retrieval results
        Retrieval_result = self.Search(result_list)
        return Retrieval_result
    
class Thinking_agent:
    # The Thinking_agent class is responsible for the following functions:
    # (1) Analyze the thinking process of the character based on the query;
    # (2) Construct "Memory Content" and "thinking" based on a segment of the Life_story.
    def __init__(self, character_infos, character_name, character_biography, personality_traits, temperature=0.0, api_base=BASE_URL, api_key=API_KEY):
        self.api_base = api_base
        self.api_key = api_key
        self.infos = character_infos
        self.name = character_name
        self.biography = character_biography
        self.personality_traits = personality_traits,
        self.temperature = temperature
        self.think = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=Model_for_agent,
            temperature = self.temperature
        )
    
    def Memory_construction(self, LifeStory_chunk):
        # Construct "Memory Content" and "thinking" based on a segment of the Life_story
        messages = []
        sys_prompt = Memory_content_construction_system_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        user_prompt = Memory_content_construction_user_prompt_template.format(
            chunk = LifeStory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate memory
        ans = self.think(messages)
        return ans.content
    
    def Thinking_Memory_construction(self, memory_chunk):
        # Generate the character's thinking about a memory chunk
        messages = []
        sys_prompt = Thinking_memory_construction_system_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        
        user_prompt = Thinking_memory_construction_user_prompt_template.format(
            chunk = memory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate thinking about the memory chunk
        ans = self.think(messages)
        return ans.content
    
    def Thinking_analysis(self, query):
        # Analyze the character's current thinking process based on the query
        messages = []
        sys_prompt = Generate_personal_think_system_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            character_biography = self.biography
        )
        user_prompt = Generate_personal_think_user_prompt_template.format(
            query = query
        )

        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        
        Thinking_result = self.think(messages)
        return Thinking_result.content

class Emotion_agent:
    # The Emotion_agent class is responsible for the following functions:
    # (1) Analyze the character's current emotional state based on the query;
    # (2) Construct "Emotion" based on a segment of the Life_story.
    def __init__(self, character_infos, character_name, personality_traits, temperature=0.0, api_base=BASE_URL, api_key=API_KEY):
        self.api_base = api_base
        self.api_key = api_key
        self.infos = character_infos
        self.name = character_name
        self.personality_traits = personality_traits
        self.temperature = temperature
        self.emotion = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=Model_for_agent,
            temperature = self.temperature
        )

    def Memory_construction(self, LifeStory_chunk):
        # Construct "Emotion Memory" based on a segment of the Life_story
        messages = []
        sys_prompt = Emotional_memory_construction_system_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        
        user_prompt = Emotional_memory_construction_user_prompt_template.format(
            chunk = LifeStory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate emotional memory
        ans = self.emotion(messages)
        return ans.content


    def Emotion_analysis(self, query):
        # Analyze the character's current emotion based on the query
        messages = []
        sys_prompt = Generate_personal_emotion_system_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )

        user_prompt = Generate_personal_emotion_user_prompt_template.format(
            query = query
        )
        
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))

        emo_result = self.emotion(messages)

        return emo_result.content
    

class Top_agent:
    # The Top_agent class is responsible for the following functions:
    # (1) Transform questions into descriptive statements and extract key content;
    # (2) Construct and maintain working memory;
    # (3) Answer queries based on working memory.
      
    def __init__(self, character_name, temperature = 0.0, api_base = BASE_URL, api_key = API_KEY):
        self.api_base = api_base
        self.api_key = api_key
        
        self.name = character_name
        self.temperature = temperature
        flag = False
        with open(Introductions_Path, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
        for introduction in introductions:
            if introduction["Name"] == self.name:
                flag = True
                self.infos = introduction['Basic_infos']
                self.personality_traits = introduction['Personality_traits']
                self.biography = introduction['Content']
        if flag == False:
            print("Can not find the information.")
            sys.exit(1)

        # Initialize three agents
        self.Thinking_Agent = Thinking_agent(character_infos=self.infos, personality_traits=self.personality_traits, character_name=self.name, character_biography=self.biography, temperature=self.temperature)
        self.Emotion_Agent = Emotion_agent(character_infos=self.infos, personality_traits=self.personality_traits, character_name=self.name, temperature=self.temperature)
        self.Memory_Agent = Memory_agent(character_name=self.name, temperature=self.temperature)
        
        self.chat = ChatOpenAI(
            openai_api_key = self.api_key,
            openai_api_base = self.api_base,
            model = Model_for_evaluation,
            temperature = self.temperature
        )
        
        if not os.path.exists(os.path.join(Memory_Directory, self.name)):
            os.mkdir(os.path.join(Memory_Directory, self.name))
            print("------ Starting to constructing the long memory for " + self.name)
            self.long_memory_construction()
            print("------ Finishing to constructing the long memory for " + self.name)
        
        

    def long_memory_construction(self):
        # Complete process function for constructing long-term memory
        
        # Initialize long_memory and index as dictionaries
        long_memory = {}
        index = {}
        
        # long_memory: {"num1": {"Memory_Summary" : "xxx" , "Memory"：{"Memory Content" : "xxx" , "Thinking" : "xxx" , "Emotion" : "xxx"}},"num2": {"Memory_Summary" : "xxx" , "Memory"：{"Memory Content" : "xxx" , "Thinking" : "xxx" , "Emotion" : "xxx"}}, ... }
        story_path = os.path.join(Story_Directory, self.name, self.name + ".txt")
        
        # Split life_story into segments
        with open(story_path, 'r', encoding='utf-8') as file:
            num = 0
            
            temp_chunk = file.readline()
            while temp_chunk:
                
                num += 1
                string_num = str(num).zfill(3)
                chunk = temp_chunk
                # Concatenate two paragraphs into one chunk
                temp_chunk = file.readline()
                chunk += temp_chunk
                
                memory_content = self.Thinking_Agent.Memory_construction(chunk)
                thinking = self.Thinking_Agent.Thinking_Memory_construction(memory_content)
                emotion = self.Emotion_Agent.Memory_construction(chunk)
                memory_summary = self.Memory_Agent.Summary(memory_content, emotion)
                memory = {"Memory Content": memory_content, "Thinking": thinking, "Emotion": emotion}
                long_memory_chunk = {"Memory_Summary" : memory_summary , "Memory" : memory}
                index[string_num] = memory_summary
                long_memory[string_num] = long_memory_chunk
                print(string_num, end="\r")  # Display progress
                print(index)
                print("--------------------------")
                print(long_memory)
                print("--------------------------")
                temp_chunk = file.readline()
        self.Memory_Agent.Add_long_memory(long_memory)
        self.Memory_Agent.Index_construction(index)
        
    def multi_turn_chat(self):
        # Multi-turn conversation, input "exit" to terminate conversation
        
        System_prompt = Naive_simulacra_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            introduction = self.biography,
        )
        current_messages = [
            SystemMessage(content=System_prompt)
        ]
        chat_history = []
        while True:
            # Get user input
            query = input()
            if query.lower() == "exit":
                break  
            memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
            thinking = self.Thinking_Agent.Thinking_analysis(query)
            emotion = self.Emotion_Agent.Emotion_analysis(query)
            
            if memory_retrieval:
                memory = str(memory_retrieval)
                memory_prompt = Multi_agent_cognitive_system_chat_memory_prompt_template.format(
                    memory = memory,
                )
                user_prompt = Multi_agent_cognitive_system_chat_prompt_template.format(
                    query = query,
                    memory = memory,
                    thinking = thinking,
                    emotion = emotion,
                    personality_traits = self.personality_traits
                )
                messages.append(SystemMessage(content=memory_prompt))
            else:
                # No related content in memory, retrieval failed
                user_prompt = Multi_agent_cognitive_system_simple_chat_prompt_template.format(
                    query = query,
                    thinking = thinking,
                    emotion = emotion,
                    personality_traits = self.personality_traits
                )
            if chat_history:
                history = "\n".join(chat_history)
                context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
                current_messages.append(HumanMessage(content=context))
            else:
                current_messages.append(HumanMessage(content="You're chatting with someone in a coffee shop."))
            current_messages.append(HumanMessage(content=user_prompt))
            agents_ans = self.chat(current_messages).content
            print(agents_ans)
            chat_history.append(query)
            chat_history.append(agents_ans)
            
        print("The conversation is over.")
        
    def bandwagon_chat(self, query, chat_history=None):
        """
        chat for bandwagon effect
        query: string
        chat_history: [AIMessage, HumanMessage]
        """
        
        System_prompt = Naive_simulacra_prompt_template.format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            introduction = self.biography,
        )
        messages = [
            SystemMessage(content=System_prompt)
        ]
        if chat_history:
            messages += chat_history
        messages.append(HumanMessage(content=query))
        memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
        thinking = self.Thinking_Agent.Thinking_analysis(query)
        emotion = self.Emotion_Agent.Emotion_analysis(query)
        if memory_retrieval:
            memory = str(memory_retrieval)
            memory_prompt = Multi_agent_cognitive_system_chat_memory_prompt_template.format(
                memory = memory
            )
            user_prompt = Multi_agent_cognitive_system_chat_prompt_template.format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
            messages.append(SystemMessage(content=memory_prompt))
        else:
            user_prompt = Multi_agent_cognitive_system_chat_prompt_template.format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        
        messages.append(SystemMessage(content=user_prompt))
        agents_ans = self.chat(messages)
        return agents_ans.content
    
    def evaluation_chat(self, query):
        # For evaluation only
        
        memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
        thinking = self.Thinking_Agent.Thinking_analysis(query)
        emotion = self.Emotion_Agent.Emotion_analysis(query)
        if memory_retrieval:
            memory = str(memory_retrieval)
            user_prompt = Multi_agent_cognitive_system_evaluation_prompt_template.format(
                memory = memory,
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        else:
            user_prompt = Multi_agent_cognitive_system_simple_evaluation_prompt_template.format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        return user_prompt


def Multi_turn_chat_with_naive_prompt(character_name):
    chat = ChatOpenAI(
        openai_api_key = API_KEY,
        openai_api_base = BASE_URL,
        model = Model_for_evaluation
    )
    flag = False
    with open(Introductions_Path, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
    if flag == False:
        print("Can not find the information.")
        sys.exit(1)
        
    System_prompt = Naive_simulacra_prompt_template.format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    current_messages = [
        SystemMessage(content=System_prompt)
    ]
    chat_history = []
    while True:
        query = input()
        if query.lower() == "exit":
            break     
        if chat_history:
            history = "\n".join(chat_history)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            current_messages.append(HumanMessage(content=context))
        else:
            current_messages.append(HumanMessage(content="You're chatting with someone in a coffee shop."))
        user_prompt = f"The one you are chatting with said:<<<{query}>>>"
        current_messages.append(HumanMessage(content=user_prompt))
        agents_ans = chat(current_messages).content
        print(agents_ans)
        chat_history.append(query)
        chat_history.append(agents_ans)
            
    print("The conversation is over.")
    

def Multi_turn_chat_with_naive_rag(character_name):
    chat = ChatOpenAI(
        openai_api_key = API_KEY,
        openai_api_base = BASE_URL,
        model = Model_for_evaluation
    )
    flag = False
    with open(Introductions_Path, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
    if flag == False:
        print("Can not find the information.")
        sys.exit(1)
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
        
    vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="openai_embed")
        
    System_prompt = Naive_simulacra_prompt_template.format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    current_messages = [
        SystemMessage(content=System_prompt)
    ]
    chat_history = []
    while True:
        query = input()

        if query.lower() == "exit":
            break  
        search_results = vectorstore.similarity_search(query, k=3)
        source_knowledge = "\n".join([x.page_content for x in search_results])
        rag_prompt = Naive_rag_simulacra_prompt_template.format(
                        source_knowledge = source_knowledge
        )
        current_messages.append(SystemMessage(content=rag_prompt))
        
        if chat_history:
            history = "\n".join(chat_history)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            current_messages.append(HumanMessage(content=context))
        else:
            current_messages.append(HumanMessage(content="You're chatting with someone in a coffee shop."))
            
        user_prompt = f"The one you are chatting with said:<<<{query}>>>"
        current_messages.append(HumanMessage(content=user_prompt))
        agents_ans = chat(current_messages).content
        print(agents_ans)
        chat_history.append(query)
        chat_history.append(agents_ans)
            
    print("The conversation is over.")
  
def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--character_name", type=str, required=True, help="Name of the character")
    parser.add_argument("--method", type=str, choices=["prompt", "rag", "macm"], required=True, help="Method of conversation: prompt, rag, or macm")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")

    # Parse arguments
    args = parser.parse_args()

    # Check if character_name is in Character_list
    if args.character_name not in Character_list:
        print(f"Error: {args.character_name} is not in the Character_list.")
        sys.exit(1)

    # Execute based on method
    if args.method == "rag":
        print(f"Starting multi-turn chat with RAG method for {args.character_name}. Type 'exit' to end the chat.")
        Multi_turn_chat_with_naive_rag(args.character_name)
    elif args.method == "macm":
        print(f"Starting multi-turn chat with MACM method for {args.character_name}. Type 'exit' to end the chat.")
        agent = Top_agent(args.character_name, temperature=args.temperature)
        memory_path = os.path.join(Memory_Directory, args.character_name, "long_memory.json")
        if not os.path.exists(memory_path):
            print("Long memory not found. Constructing long memory...")
            agent.long_memory_construction()
            print("Long memory construction completed.")
        agent.multi_turn_chat()
    else:
        print(f"Starting multi-turn chat with prompt method for {args.character_name}. Type 'exit' to end the chat.")
        Multi_turn_chat_with_naive_prompt(args.character_name)

if __name__ == "__main__":
    main()