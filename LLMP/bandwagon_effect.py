import argparse
from Config.config import *
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from prompt_templates import Bandwagon_effect_system_prompt_template, Bandwagon_effect_user_prompt_template, Bandwagon_effect_controlled_user_prompt_template
from multi_agent_cognitive_mechanism import Top_agent

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


def MACM_bandwagon_effect(character_name, temperature = 0.0, control=True):
    """
    Simulates the bandwagon effect using multi-agent cognitive mechanisms.
    """
    critical_subject = Top_agent(character_name, temperature=temperature)
    System_prompt = Bandwagon_effect_system_prompt_template.format(
        standard_len = experiment_config[0][0],
        len_1 = experiment_config[0][1],
        len_2 = experiment_config[0][2],
        len_3 = experiment_config[0][3],
        group_response = experiment_config[0][5]
    )
    answer = critical_subject.bandwagon_chat(query=System_prompt)
    chat_history = [
        HumanMessage(content=System_prompt),
        AIMessage(content=answer)
    ]
    for i in range(1, len(experiment_config)):
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
        answer = critical_subject.Chat(query=User_prompt, chat_history=chat_history)
        chat_history.append(HumanMessage(content=User_prompt))
        chat_history.append(AIMessage(content=answer))
    print(chat_history)
        

def main():
    MACM_bandwagon_effect("Erica Walker", control=False)
def main():
    parser = argparse.ArgumentParser(description="Simulate the bandwagon effect using multi-agent cognitive mechanisms.")
    parser.add_argument('--character_name', type=str, required=True, help='Name of the character')
    parser.add_argument('--control', action='store_true', help='Use control group')
    
    args = parser.parse_args()
    # Check if character_name is in Character_list
    if args.character_name not in Character_list:
        print(f"Error: {args.character_name} is not in the Character_list.")
        sys.exit(1)
    MACM_bandwagon_effect(args.character_name, control=args.control)

if __name__ == "__main__":
    main()
