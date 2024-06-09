"""
generate virtual characters from a set of candidate attribute pools
"""

import json
import random
import os
from faker import Faker
from Config.config import Current_Directory, Attributes_Directory

fake = Faker()

def init_candidate_pools():
    candidate_pools = {} # {string:[] or string:{}}
    
    # Read occupations from file and store them in the occupation pool
    with open(os.path.join(Attributes_Directory, "occupations.txt"), "r", encoding="UTF-8") as file:
        occupations = file.readlines()
        for i in range(len(occupations)- 1):
            occupations[i] = occupations[i][:-1]
        candidate_pools["occupation_pool"] = occupations
        
    with open(os.path.join(Attributes_Directory, "traits.txt"), "r", encoding="UTF-8") as file:
        trait_pool = {} # {string:[[],[]]}
        traits = file.readlines()

        for idx, data in enumerate(traits):
            rank = (idx % 9)
            if rank == 0:
                trait = data.replace("\n", "")
                trait_pool[trait] = []
            else:
                des_sens_for_dimension = data.split(". ")
                des_sens_for_dimension[-1] = des_sens_for_dimension[-1][:-2]
                trait_pool[trait].append(des_sens_for_dimension)
        candidate_pools["trait_pool"] = trait_pool
    
    with open(os.path.join(Attributes_Directory, "hobbies.txt"), "r", encoding="UTF-8") as file:
        hobbies = file.readlines()
        for i in range(len(hobbies)- 1):
            hobbies[i] = hobbies[i][:-1]
        candidate_pools["hobby_pool"] = hobbies
    
    with open(os.path.join(Attributes_Directory, "family.txt"), "r", encoding="UTF-8") as file:
        families = file.readlines()
        for i in range(len(families)- 1):
            families[i] = families[i][:-1]
        candidate_pools["family_pool"] = families
    
    with open(os.path.join(Attributes_Directory, "education.txt"), "r", encoding="UTF-8") as file:
        educations = file.readlines()
        for i in range(len(educations)- 1):
            educations[i] = educations[i][:-1]
        candidate_pools["education_pool"] = educations  
          
    with open(os.path.join(Attributes_Directory, "short-term_goals.txt"), "r", encoding="UTF-8") as file:
        short_goals = file.readlines()
        for i in range(len(short_goals)- 1):
            short_goals[i] = short_goals[i][:-1]
        candidate_pools["short-term_goals_pool"] = short_goals
        
    with open(os.path.join(Attributes_Directory, "long-term_goals.txt"), "r", encoding="UTF-8") as file:
        long_goals = file.readlines()
        for i in range(len(long_goals)- 1):
            long_goals[i] = long_goals[i][:-1]
        candidate_pools["long-term_goals_pool"] = long_goals 
    
    print(json.dumps(candidate_pools, sort_keys=True, indent=4, separators=(',', ': ')))
    
    return candidate_pools

def generate_name(gender):
    if gender == "female":
        return fake.name_female()
    elif gender == "male":
        return fake.name_male()
    elif gender == "non-binary":
        return fake.name_nonbinary()
    else:
        return fake.name()  # Generate arbitrary names by default

def generate_traits(trait_pool):
    traits = []
    traits_text = ""
    Jungian_types = ["Ne", "Ni", "Se", "Si", "Te", "Ti", "Fe", "Fi"]
    prior_types = ["Ne", "Ni", "Se", "Si"]
    second_prior_types = ["Te", "Ti", "Fe", "Fi"]
    traits.append(random.sample(prior_types, 1)[0])
    traits.append(random.sample(second_prior_types, 1)[0])
    remain_types = [x for x in Jungian_types if x not in traits]
    random.shuffle(remain_types)
    traits += remain_types
    for idx, trait in enumerate(traits):
        if idx in [0, 7]:
            dim_sens = trait_pool[trait][idx]
            des_sentence = random.sample(dim_sens, 4)  #[0] + ". "
            traits_text = traits_text  + ". - ".join(des_sentence)
            traits_text += ". "
        if idx in [1, 6]:
            dim_sens = trait_pool[trait][idx]
            des_sentence = random.sample(dim_sens, 3)  #[0] + ". "
            traits_text = traits_text  + ". - ".join(des_sentence)
            traits_text += ". "
        if idx in [2, 5]:
            dim_sens = trait_pool[trait][idx]
            des_sentence = random.sample(dim_sens, 2)  #[0] + ". "
            traits_text = traits_text  + ". - ".join(des_sentence)
            traits_text += ". "
        if idx in [3, 4]:
            dim_sens = trait_pool[trait][idx]
            des_sentence = random.sample(dim_sens, 1)  #[0] + ". "
            traits_text = traits_text  + ". - ".join(des_sentence)
            traits_text += ". "
    traits_text = "- " + traits_text
    return traits_text

def generate_characters(candidate_pools, character_number):
    
    infos = [] # [{}, {}, {}]
    for i in range(character_number):
        info = {}
        info["ID"] = str(i).zfill(3)
        info["Gender"] = random.sample(["female", "male", "non-binary"], 1)[0]

        info["Name"] = generate_name(info["Gender"])
        info["Age"] = random.sample(range(20, 56), 1)[0]
        info["Date_of_birth"] = fake.date_of_birth(minimum_age = info["Age"], maximum_age = info["Age"]).strftime('%Y-%m-%d')
        info["Photo"] = "NULL" # # Placeholder for photo
        info["Occupation"] = random.sample(candidate_pools["occupation_pool"], 1)[0]
        
        info["Traits"] = generate_traits(candidate_pools["trait_pool"])
        
        info["Hobbies"] = ", ".join(random.sample(candidate_pools["hobby_pool"], 3))
        info["Family"] = random.sample(candidate_pools["family_pool"], 1)[0]
        info["Education"] = random.sample(candidate_pools["education_pool"], 1)[0]
        info["Short-term_goals"] = ", ".join(random.sample(candidate_pools["short-term_goals_pool"], 3))
        info["Long-term_goal"] = random.sample(candidate_pools["long-term_goals_pool"], 1)[0]

        infos.append(info)
    
    return infos
        


def main():
    os.chdir(Current_Directory)
    
    character_number = 100
    
    candidate_pools = init_candidate_pools()
    
    infos = generate_characters(candidate_pools, character_number)
    
    with open("Candidate_character_profiles.json", "w", encoding="UTF-8") as file:
        json.dump(infos, file, ensure_ascii=False, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    main()
