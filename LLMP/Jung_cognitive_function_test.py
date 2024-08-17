import argparse
import sys
import json
import random
import math
import subprocess
import os
from colorama import Fore, Style
from Config.config import *
from scipy.stats import kendalltau, spearmanr

rank_to_score = {
    0 : 4,
    1 : 3,
    2 : 2,
    3 : 1,
    4 : -1,
    5 : -2,
    6 : -3,
    7 : -4
}
Jung_cognitive_function = {
    "Mary Jones" : ['Se', 'Ti', 'Fi', 'Ni', 'Te', 'Si', 'Ne', 'Fe'],
    "Tami Clark" : ['Se', 'Fi', 'Ne', 'Ti', 'Ni', 'Fe', 'Te', 'Si'],
    "Sara Ochoa" : ['Ni', 'Ti', 'Te', 'Fe', 'Fi', 'Ne', 'Se', 'Si'],
    "Michael Miller" : ['Si', 'Fi', 'Fe', 'Ne', 'Ti', 'Ni', 'Se', 'Te'],
    "Haley Collins" : ['Si', 'Te', 'Ne', 'Ti', 'Se', 'Fe', 'Ni', 'Fi'],
    "James Jones" : ['Ne', 'Fi', 'Fe', 'Ni', 'Se', 'Ti', 'Te', 'Si'],
    "Kevin Kelly" : ['Si', 'Ti', 'Ni', 'Ne', 'Fe', 'Te', 'Fi', 'Se'],
    "Erica Walker" : ['Se', 'Fe', 'Ne', 'Fi', 'Ni', 'Si', 'Ti', 'Te'],
    "Leslie Nichols" : ['Se', 'Te', 'Ne', 'Ni', 'Si', 'Fe', 'Ti', 'Fi'],
    "Robert Scott" : ['Ne', 'Ti', 'Si', 'Fe', 'Se', 'Fi', 'Te', 'Ni'],
    "Marsh Zhaleh" : ['Ne', 'Fe', 'Ni', 'Te', 'Ti', 'Fi', 'Se', 'Si']
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
def get_latest_folder_path(directory_path):
    try:
        # Get all entries (files and directories) in the specified directory
        all_entries = os.listdir(directory_path)
        
        # Filter out the entries to get only directories
        folders = [entry for entry in all_entries if os.path.isdir(os.path.join(directory_path, entry))]

        if not folders:
            return None  # Return None if there are no directories
        
        # Create full paths for each directory
        folder_paths = [os.path.join(directory_path, folder) for folder in folders]

        # Find the directory with the latest creation time
        latest_folder = max(folder_paths, key=os.path.getctime)
        
        # Return the path of the latest directory
        return latest_folder
    
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
        return None
    except PermissionError:
        print(f"Permission denied to access the directory '{directory_path}'.")
        return None            

def Test_item_extraction(character_name, ratio=1.0):
    
    extract_nums = math.floor(ratio * 10)
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
    test_items = [] # each item is {"trait":"str", "type": "str", "score": from 4 to -4}
    for trait in trait_pool.keys():
        for idx, trait_per_rank in enumerate(trait_pool[trait]):
            extract_traits = random.sample(trait_per_rank, extract_nums)
            for description in extract_traits:
                test_item = {
                    "type": "Jung_cognitive_function_test",
                    "question": description,
                    "function": trait,
                    "score": rank_to_score[idx],
                    "options": [""],
                    "answer": [trait, str(rank_to_score[idx])] # save for score calculation
                }
                test_items.append(test_item)
    with open(os.path.join(Question_Directory, character_name, "Jung_cognitive_function_test.json"), "w", encoding="UTF-8") as file:
        json.dump(test_items, file, ensure_ascii=False, indent=4, separators=(',', ': '))

def hamming_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")
    
    return sum(el1 != el2 for el1, el2 in zip(list1, list2))

def count_inversions(list1, list2):
    index_map = {value: i for i, value in enumerate(list1)}
    inversions = 0
    
    for i in range(len(list2)):
        for j in range(i + 1, len(list2)):
            if index_map[list2[i]] > index_map[list2[j]]:
                inversions += 1
    
    return inversions

def Jung_cognitive_function_test(character_name, method="prompt", if_extract=False, if_rerun=False, ratio=1.0):
    """
    Conduct Jung cognitive function on base prompt-/rag-/multi-agent cognitive mechanisms-based simulacra.
    """
    print(Fore.RED + f"Starting Jung cognitive function test on " + method + "-based simulacra." + Style.RESET_ALL)
    print(Fore.RED + f"Current character: " + character_name + Style.RESET_ALL)
    
    if if_extract:
        Test_item_extraction(character_name, ratio)
        print(Fore.RED + f"Finish test item extraction." + Style.RESET_ALL)
        
    if if_rerun:
        command = [
            "python", 
            os.path.join(Current_Directory, "opencompass","run.py"), 
            os.path.join(Current_Directory, "opencompass", "configs", "datasets", "LLMP", "LLMP_gen_single.py"), 
            "-w", 
            os.path.join(Current_Directory, "Outputs", "Jung_test", character_name, method)
        ]
        print(Fore.RED + f"Starting rerun." + Style.RESET_ALL)
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read the subprocess output line by line
        for line in process.stdout:
            print(line, end="") 

        stderr_output = process.stderr.read()
        if stderr_output:
            print(Fore.RED + "Error:" + Style.RESET_ALL)
            print(stderr_output)

        # Wait for the subprocess to finish
        process.stdout.close()
        process.stderr.close()
        process.wait()
        
        print(Fore.RED + f"Finishing rerun." + Style.RESET_ALL)
    
    
    folder_path = get_latest_folder_path(os.path.join(Output_Directory, "Jung_test", character_name, method))
    prediction_path = os.path.join(folder_path, "predictions", character_name, "LLMP_" + character_name + "_Jung_cognitive_function_test.json")
    with open(prediction_path, "r", encoding="UTF-8") as file:
        predictions = json.load(file)
    function_scores = {
        "Ne": 0,
        "Ni": 0,
        "Fe": 0,
        "Fi": 0,
        "Se": 0,
        "Si": 0,
        "Te": 0,
        "Ti": 0
    }
    for item_idx in predictions.keys():
        answer = predictions[item_idx]["prediction"].lower()
        function = predictions[item_idx]["gold"][0]
        score = int(predictions[item_idx]["gold"][1])
        if "yes" in answer:
            function_scores[function] += score
        else:
            continue
    sorted_scores = sorted(function_scores.items(), key=lambda item: item[1], reverse=True)
    print(Fore.RED + f"The result of Jung cognitive function test for {character_name} with {method} is: \n{sorted_scores}" + Style.RESET_ALL)
    sorted_keys = [key for key, value in sorted_scores]
    print(Fore.RED + f"The ranking of Jung cognitive function test for {character_name} with {method} is: \n{sorted_keys}" + Style.RESET_ALL)
    gt = Jung_cognitive_function[character_name]
    kendall_rank_correlation, _ = kendalltau(sorted_keys, gt)
    print(Fore.RED + f"The correlation of Jung cognitive function test for {character_name} with {method} is: \n{kendall_rank_correlation}" + Style.RESET_ALL)
    

    
    
        

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Conduct Jung cognitive function on base prompt-/rag-/multi-agent cognitive mechanisms-based simulacra.")

    # Add arguments
    parser.add_argument("--character_name", type=str, required=True, help="Name of the character")
    parser.add_argument("--method", type=str, choices=["prompt", "rag", "macm"], required=True, help="Method of conversation: prompt, rag, or macm")
    parser.add_argument("--if_extract", action='store_true', help="Test items extraction")
    parser.add_argument("--if_rerun", action='store_true', help="Test rerun")
    parser.add_argument("--ratio", type=float, default=0.4, help="Ratio for test items extraction")

    # Parse arguments
    args = parser.parse_args()

    # Check if character_name is in Character_list
    if args.character_name not in Character_list:
        print(f"Error: {args.character_name} is not in the Character_list.")
        sys.exit(1)

    # Execute based on method

    Jung_cognitive_function_test(args.character_name, args.method, args.ratio, args.if_extract, args.if_rerun)




if __name__ == "__main__":
    main()
