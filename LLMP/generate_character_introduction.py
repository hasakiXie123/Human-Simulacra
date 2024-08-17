"""
Generate character introductions based on character profiles
"""
# # Read Candidate_character_profiles.json, for each profile, reassemble into a prompt, receive gpt's response, save as Candidate_character_introductions.json

import os
import json
from openai import OpenAI
from prompt_templates import Generate_introductions_prompt_template, Profile_prompt_template
from Config.config import Current_Directory, Profiles_Path, OPENAI_API_KEY, OPENAI_BASE_URL, Model_for_data

def main():
    os.chdir(Current_Directory)
    
    client = OpenAI(
        base_url = OPENAI_BASE_URL,
        api_key = OPENAI_API_KEY
    )
    
    with open("Candidate_character_profiles.json", "r", encoding="UTF-8") as file:
        profiles = json.load(file)
        introductions = []
        for profile in profiles:
            introduction = {}
            system_prompt = Generate_introductions_prompt_template.format(
                character_name = profile['Name']
            )
            
            user_prompt = Profile_prompt_template.format(
                character_name = profile['Name'], character_age = profile['Age'],
                character_gender = profile['Gender'], character_dob = profile['Date_of_birth'],
                character_occupation = profile['Occupation'], character_trait = profile['Traits'],
                character_hobby = profile['Hobbies'], character_family = profile['Family'],
                character_education = profile['Education'], character_s_goal = profile['Short-term_goals'],
                character_l_goal = profile['Long-term_goal']
            )
                
            response = client.chat.completions.create(
                model = Model_for_data,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # frequency_penalty=1.0,
                temperature=0.5,
                # top_p=0.95
            )
            # Create the introduction dictionary
            introduction["ID"] = profile["ID"]
            introduction["Name"] = profile['Name']
            introduction["Personality_traits"] = profile['Traits']
            temp_str = user_prompt.split(":\n")[1]
            start = temp_str.find("Personality Traits:")
            end = temp_str.find("Hobbies:", start)
            Basic_infos = temp_str[:start] + temp_str[end:]
            introduction["Basic_infos"] = Basic_infos
            introduction["Extra"] = {}
            introduction['Content'] = response.choices[0].message.content
            introductions.append(introduction)
            
            print("------ Finishing " + introduction["Name"])
    
    with open("Candidate_character_introductions.json", "w", encoding="UTF-8") as file:
        json.dump(introductions, file, ensure_ascii=False, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    main()
