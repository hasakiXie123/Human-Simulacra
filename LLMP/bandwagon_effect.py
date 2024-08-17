import argparse
import sys
import os
from colorama import Fore, Style
from Config.config import *
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from prompt_templates import Bandwagon_effect_system_prompt_template, Bandwagon_effect_controlled_system_prompt_template, Bandwagon_effect_user_prompt_template, Bandwagon_effect_controlled_user_prompt_template
from multi_agent_cognitive_mechanism import Top_agent, Bandwagon_chat_with_naive_prompt, Bandwagon_chat_with_naive_rag, Bandwagon_chat_with_blank_model

### [standard line, comparison line 1, comparison line 2, comparison line 3, correct response, group response]
experiment_config = [
    [10, 8.75, 10, 8, 2, 2],
    [2, 2, 1, 1.5, 1, 1],
    [3, 3.75, 4.25, 3, 3, 1],
    [5, 5, 4, 6.5, 1, 2],
    [4, 3, 5, 4, 3, 3],
    [3, 3.75, 4.25, 3, 3, 2],
    [8, 6.25, 8, 6.75, 2, 3],
    [5, 5, 4, 6.5, 1, 3],
    [8, 6.25, 8, 6.75, 2, 1],
    [10, 8.75, 10, 8, 2, 2],
    [2, 2, 1, 1.5, 1, 1],
    [3, 3.75, 4.25, 3, 3, 1],
    [5, 5, 4, 6.5, 1, 2],
    [4, 3, 5, 4, 3, 3],
    [3, 3.75, 4.25, 3, 3, 2],
    [8, 6.25, 8, 6.75, 2, 3],
    [5, 5, 4, 6.5, 1, 3],
    [8, 6.25, 8, 6.75, 2, 1],
]

def colorful_print(chat_history):
    print(Fore.RED + f"Below is the experiment log" + Style.RESET_ALL)
    for message in chat_history:
        if isinstance(message, AIMessage):
            print("-------------------------------")
            print(Fore.GREEN + message.content + Style.RESET_ALL)
        elif isinstance(message, HumanMessage):
            # print(message.content)
            pass
        else:
            ## system
            # print(Fore.RED + message.content + Style.RESET_ALL)
            pass
            

def Bandwagon_effect(character_name, method="macm", temperature = 0.0, control=True):
    """
    Simulates the bandwagon effect.
    """
    print(Fore.RED + f"Starting Simulates the bandwagon effect using " + method + " mechanisms." + Style.RESET_ALL)
    print(Fore.RED + f"Current character: " + character_name + Style.RESET_ALL)
    
    System_prompt = Bandwagon_effect_system_prompt_template.format(
        standard_len = experiment_config[0][0],
        len_1 = experiment_config[0][1],
        len_2 = experiment_config[0][2],
        len_3 = experiment_config[0][3],
        group_response = experiment_config[0][5]
    )
    if control:
        System_prompt = Bandwagon_effect_controlled_system_prompt_template.format(
        standard_len = experiment_config[0][0],
        len_1 = experiment_config[0][1],
        len_2 = experiment_config[0][2],
        len_3 = experiment_config[0][3]
    )
    if method == "macm":
        critical_subject = Top_agent(character_name, temperature=temperature)
        memory_path = os.path.join(Memory_Directory, character_name, "long_memory.json")
        if not os.path.exists(memory_path):
            print("Long memory not found. Constructing long memory...")
            critical_subject.long_memory_construction()
            print("Long memory construction completed.")
        
        answer = critical_subject.bandwagon_chat(query=System_prompt)
    elif method == "prompt":
        answer = Bandwagon_chat_with_naive_prompt(character_name, System_prompt)
    elif method == "rag":
        answer = Bandwagon_chat_with_naive_rag(character_name, System_prompt)
    else:
        answer = Bandwagon_chat_with_naive_prompt(character_name, System_prompt)
    chat_history = [
        HumanMessage(content=System_prompt),
        AIMessage(content=answer)
    ]
    
    for i in range(1, len(experiment_config)):
        print(Fore.RED + f"Conducting the " + str(i) +"th " + "experiment." + Style.RESET_ALL)
        if control:
        # Control group: Results are not influenced by group pressure
            User_prompt = Bandwagon_effect_controlled_user_prompt_template.format(
                standard_len = experiment_config[i][0],
                len_1 = experiment_config[i][1],
                len_2 = experiment_config[i][2],
                len_3 = experiment_config[i][3]
            )
        else:
            # Experimental group: Results are influenced by group pressure
            User_prompt = Bandwagon_effect_user_prompt_template.format(
                standard_len = experiment_config[i][0],
                len_1 = experiment_config[i][1],
                len_2 = experiment_config[i][2],
                len_3 = experiment_config[i][3],
                group_response = experiment_config[i][5]
            )
        if method == "macm":
            answer = critical_subject.bandwagon_chat(query=User_prompt, chat_history=chat_history)
        elif method == "prompt":
            answer = Bandwagon_chat_with_naive_prompt(character_name, User_prompt, chat_history)
        else:
            answer = Bandwagon_chat_with_blank_model(User_prompt, chat_history)
        chat_history.append(HumanMessage(content=User_prompt))
        chat_history.append(AIMessage(content=answer))
    colorful_print(chat_history)
        

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Simulate the bandwagon effect using base prompt/rag/multi-agent cognitive mechanisms.")

    # Add arguments
    parser.add_argument("--character_name", type=str, required=True, help="Name of the character")
    parser.add_argument("--method", type=str, choices=["prompt", "rag", "macm", "none"], required=True, help="Method of conversation: prompt, rag, none or macm")
    parser.add_argument('--control', action='store_true', help='Use control group')

    # Parse arguments
    args = parser.parse_args()

    # Check if character_name is in Character_list
    if args.character_name not in Character_list:
        print(f"Error: {args.character_name} is not in the Character_list.")
        sys.exit(1)

    # Execute based on method
    Bandwagon_effect(args.character_name, args.method, control=args.control)



if __name__ == "__main__":
    main()
