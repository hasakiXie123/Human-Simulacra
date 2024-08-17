"""
Generate the life story of character in a step manner
"""
import os
# Disable parallelism in tokenizers to avoid potential issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
from transformers import pipeline
from openai import OpenAI
from prompt_templates import *
from Config.config import *
from utils import chunk_sorting, print_statistics
import argparse

def assemble_context(life_story, chunk_idx):
    """
    Assemble the context for a given chunk index in the life story.
    
    Parameters:
    life_story (list): The list containing the life story.
    chunk_idx (int): The index of the chunk to be processed.
    
    Returns:
    str: The compressed draft of the life story context.
    """

    # ensure that indexes do not result in negative numbers or exceed the list length
    start = max(0, chunk_idx - Window_size)
    end = min(len(life_story), chunk_idx + Window_size + 1)
    draft = life_story[start:end]
    
    # compressing the paragraph above and below
    summarizer = pipeline("summarization", model=os.path.join(LLM_Directory, "led-large-book-summary"))
    if start != 0:
        article = life_story[:start]
        article = "\n".join(article)
        w_count = len(article.split(" "))
        if w_count > 500:
            summary = summarizer(article, max_length=500, no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, repetition_penalty=3.5, num_beams=4, early_stopping=True)[0]['summary_text']
        else:
            summary = article
        draft.insert(0, summary)
    if end < len(life_story):
        article = life_story[end:]
        article = "\n".join(article)
        w_count = len(article.split(" "))
        if w_count > 500:
            summary = summarizer(article, max_length=500, no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, repetition_penalty=3.5, num_beams=4, early_stopping=True)[0]['summary_text']
        else:
            summary = article
        draft.append(summary)
        
    draft = "\n\n".join(draft)
    return draft
    

def generate_all(introductions):
    """
    Generate life stories for all characters in the given introductions.
    
    Parameters:
    introductions (list): A list of dictionaries containing character introductions.
    """
    for intro in introductions:
        print("------ Starting to generate the life story of " + intro["Name"])
        generate_one_gpt(intro)
        print("------ Finishing to generate the life story of " + intro["Name"])
    
    
def generate_one_gpt(introduction):
    """
    Generate the life story of a single character using GPT.
    
    Parameters:
    introduction (dict): A dictionary containing the introduction of the character.
    """
    client = OpenAI(
            base_url = OPENAI_BASE_URL,
            api_key = OPENAI_API_KEY
    )
    character_name = introduction['Name']
    self_Story_Path = os.path.join(Story_Directory, character_name)
    if not os.path.exists(self_Story_Path):
        os.makedirs(self_Story_Path)
        
    # Initialize life story and iteration start point
    life_story = introduction["Content"]
    Iteration_start = 0
    if Continued:
        with open(Continued_story_path, "r", encoding="UTF-8") as file:
            story = file.readlines()
        life_story = "\n".join(story)
        Iteration_start = int(re.search(r"-(\d+)\.", Continued_story_path).group(1)) + 1
    limited_generation = False # Flag to switch to limited context generation when needed
    
    for i in range(Iteration_start, Iteration_for_story):
        print("------ Entering iteration " + str(i))
        
        if i == Iteration_start:
            life_story = life_story.split("\n")
            life_story = list(filter(lambda x: x not in ["", " ", "  ", "\n"], life_story))
        print_statistics(life_story)
        
        expandable_chunk_idx, expandable_chunk_score = chunk_sorting(life_story)
        expandable_chunk_idx = expandable_chunk_idx[0]
        expandable_chunk_score = expandable_chunk_score[0]
        print("------ Paragraph {idx} is about to be expanded, with a score of {score}.".format(
            idx = expandable_chunk_idx,
            score = expandable_chunk_score
        ))
        
        if expandable_chunk_score < Iteration_threshold: # no need for further expanding
            print("------ Generation stopped, score: " + str(expandable_chunk_score))
            break
        
        system_prompt = Generate_life_story_system_prompt_template.format(
                character_name = character_name
        )
        if limited_generation:
            draft = assemble_context(life_story, expandable_chunk_idx)
        else:
            draft = "\n\n".join(life_story)
        character_infos = introduction["Basic_infos"]
        if introduction["Extra"]:
            # Convert the Extra dictionary to a string format
            extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
            character_infos = character_infos.strip('"\n') 
            # Append the Extra information to the Basic_infos string and restore the original format
            character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
        user_prompt = Generate_life_story_user_prompt_template.format(
            basic_information = character_infos,
            draft = draft,
            personality_traits = introduction["Personality_traits"],
            paragraph = life_story[expandable_chunk_idx],
        )
        response = client.chat.completions.create(
            model = Model_for_data,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            frequency_penalty=1.0,
#            temperature=0.7,
            top_p=0.95
        )
        prompt_tokens = response.usage.prompt_tokens
        print("------ Token Usage: " + str(prompt_tokens))
        if prompt_tokens >= Context_length - 100: # turn to limited generation
            limited_generation = True 
            print("------ Turning to limited generation at iteration " + str(i + 1))
        
        # Replace original paragraph with expanded paragraphs
        expanded_paragraphs = response.choices[0].message.content
        expanded_paragraphs = expanded_paragraphs.split("\n")
        life_story[expandable_chunk_idx:expandable_chunk_idx + 1] = expanded_paragraphs
        life_story = list(filter(lambda x: x not in ["", " ", "  ", "\n"], life_story))
        
        
        with open(os.path.join(self_Story_Path, character_name + "-" + str(i) +".txt"), "w", encoding="UTF-8") as file:
            modified_story = [s + '\n' for s in life_story]
            file.writelines(modified_story)
            
        print("------ Finishing iteration " + str(i))
        

    print_statistics(life_story)
         

def main():
    parser = argparse.ArgumentParser(description='Generate the life stories of virtual characters in a step manner.')
    parser.add_argument('--character_name', type=str, help='The name of the character')
    args = parser.parse_args()
    
    os.chdir(Current_Directory)
    
    with open(Introductions_Path, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    
    if args.character_name:
        for intro in introductions:
            if intro["Name"] == args.character_name:
                generate_one_gpt(intro)
                break
    else:
        generate_all(introductions)

if __name__ == "__main__":
    main()
