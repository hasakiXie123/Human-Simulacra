import torch
import random
import os
import sys
import numpy as np
import json 
import itertools
import torch
from colorama import Fore, Style

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from Config.config import Current_Directory, LLM_Directory, Profiles_Path, Attributes_Directory

def calculate_redundancy(chunks):
    # Load the sentence transformer model
    model = SentenceTransformer(os.path.join(LLM_Directory, "all-mpnet-base-v2"))
    chunk_embeddings = model.encode(chunks)
    scores = util.cos_sim(chunk_embeddings, chunk_embeddings)
    # Convert similarity scores to redundancy scores
    scores = 1.0 - scores
    scores = torch.mean(scores, dim=0)
    return scores.tolist()

def calculate_elaborateness(chunks):
    summarizer = pipeline("summarization", model=os.path.join(LLM_Directory, "Falconsai_text_summarization"))
    token_count = [summarizer.tokenizer(chunk, return_tensors='pt').input_ids.size(1) for chunk in chunks]
    summaries = [summarizer(chunk, max_length=token_count[idx], do_sample=False)[0]['summary_text'] for idx, chunk in enumerate(chunks)]

    model = SentenceTransformer(os.path.join(LLM_Directory, "all-mpnet-base-v2"))
    # Encode the summary and chunks into embeddings
    summary_embeddings = model.encode(summaries)
    chunk_embeddings = model.encode(chunks)
    # Calculate cosine similarity between chunk embeddings and summary embeddings
    scores = util.cos_sim(chunk_embeddings, summary_embeddings) 
    scores = np.diag(scores)
    return scores.tolist()
    
def calculate_importance(chunks, article):
    summarizer = pipeline("summarization", model=os.path.join(LLM_Directory, "led-large-book-summary"))
    tokens = summarizer.tokenizer(article, return_tensors='pt')
    token_count = tokens.input_ids.size(1)
    summary = summarizer(article, max_length=token_count - 1, min_length=0, no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, repetition_penalty=3.5, num_beams=4, early_stopping=True)[0]['summary_text']
    model = SentenceTransformer(os.path.join(LLM_Directory, "all-mpnet-base-v2"))
    article_embeddings = model.encode(summary)
    chunk_embeddings = model.encode(chunks)
    # Calculate cosine similarity between article summary embeddings and chunk embeddings
    scores = util.cos_sim(article_embeddings, chunk_embeddings) 
    scores = scores[0].tolist()
    return scores       


def chunk_sorting(chunks, top_n=1, alpha_1=80.0, alpha_2=100.0, alpha_3=120.0):
    """
    Sort chunks based on their importance, elaborateness, and redundancy.
    
    Parameters:
    - chunks: List of text chunks to be sorted
    - top_n: Number of top chunks to return
    - alpha_1: Weight for importance
    - alpha_2: Weight for elaborateness
    - alpha_3: Weight for redundancy
    
    Returns:
    - expandable_chunk_idxs: Indices of top chunks
    - expandable_chunk_scores: Scores of top chunks
    """
    # Combine all chunks into a single article
    article = " ".join(chunks)
    chunks_score = []
    expandable_chunk_idxs = []
    expandable_chunk_scores = []
    # Calculate importance, elaborateness, and redundancy scores for each chunk
    importance = calculate_importance(chunks, article) 
    elaborateness = calculate_elaborateness(chunks) 
    redundancy = calculate_redundancy(chunks) 
    # Calculate a combined score for each chunk
    for i in range(len(chunks)):
        score = alpha_1 * importance[i] + alpha_2 * elaborateness[i] + alpha_3 * redundancy[i]
        chunks_score.append((i, score, importance[i], elaborateness[i], redundancy[i]))        
        
    chunks_score.sort(key=lambda x:x[1], reverse=True)
    
    if top_n == 1:
        # Add some randomness to the selection of the top chunk
        random_idx = random.randint(0, 4)
        if len(chunks) > 20:
            random_idx = random.randint(0, len(chunks) // 10)
        expandable_chunk_idxs.append(chunks_score[random_idx][0])
        expandable_chunk_scores.append(chunks_score[random_idx][1])
    
    for i in range(top_n):
        expandable_chunk_idxs.append(chunks_score[i][0])
        expandable_chunk_scores.append(chunks_score[i][1])
        
    return expandable_chunk_idxs, expandable_chunk_scores

def print_statistics(life_story):
    # Calculate the number of paragraphs and words in the life story
    p_count = len(life_story)
    story = " ".join(life_story)
    w_count = sum(1 for w in story.split(" ") if w.strip() not in ['', "\n", "\n\n"])
    
    print("------ The story contains {p_count} paragraphs and {w_count} words.".format(
        p_count = p_count,
        w_count = w_count
    ))



def calculate_similarity_between_traits():

    model = SentenceTransformer(os.path.join(LLM_Directory, "all-mpnet-base-v2"))

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

    ## the semantic textual similarity within one rank in one tendency
    similarity_within_one_trait_one_rank = {}
    avg_similarity_within_one_trait_one_rank = 0.0
    count = 0
    for trait in trait_pool.keys():
        descriptions_per_trait = trait_pool[trait]
        for idx, descriptions_per_rank in enumerate(descriptions_per_trait):
            embeddings = model.encode(descriptions_per_rank)
            similarities = util.cos_sim(embeddings, embeddings)
            mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
            non_diag_elements = similarities[mask]
            mean_similarity = non_diag_elements.mean()
            key = trait + " at rank: " + str(idx + 1)
            similarity_within_one_trait_one_rank[key] = mean_similarity
            avg_similarity_within_one_trait_one_rank += mean_similarity
            count += 1
            
    avg_similarity_within_one_trait_one_rank /= count # 0.3037
    print("similarity within one trait one rank: ", similarity_within_one_trait_one_rank)
    print(Fore.RED + "average similarity within one trait one rank: " + str(avg_similarity_within_one_trait_one_rank) + Style.RESET_ALL)
    print()

    ## the semantic textual similarity within one tendency
    similarity_within_one_trait = {}
    avg_similarity_within_one_trait = 0.0
    count = 0
    for trait in trait_pool.keys():
        descriptions_per_trait = trait_pool[trait]
        embeddings_per_rank = []
        for idx, descriptions_per_rank in enumerate(descriptions_per_trait):
            embeddings = model.encode(descriptions_per_rank)
            embeddings_per_rank.append(embeddings)
        combinations = list(itertools.combinations(embeddings_per_rank, 2))
        temp = 0.0
        for combo in combinations:
            similarities = util.cos_sim(combo[0], combo[1])
            mean_similarity = similarities.mean()
            avg_similarity_within_one_trait += mean_similarity
            temp += mean_similarity
            count += 1
        similarity_within_one_trait[trait] = temp / len(combinations)
        
        
    avg_similarity_within_one_trait /= count # 0.3169
    print("similarity within one trait: ", similarity_within_one_trait)
    print(Fore.RED + "average similarity within one trait: " + str(avg_similarity_within_one_trait) + Style.RESET_ALL)
    print()

    ## the semantic textual similarity between eight tendency
    similarity_within_eight_trait = {}
    avg_similarity_within_eight_trait = 0.0
    count = 0
    embeddings = []
    traits = []
    for trait in trait_pool.keys():
        descriptions_per_trait = trait_pool[trait]
        flattened_descriptions = [item for sublist in descriptions_per_trait for item in sublist]
        embeddings_per_trait = model.encode(flattened_descriptions)
        embeddings.append(embeddings_per_trait)
        traits.append(trait)
    combinations = list(itertools.combinations(embeddings, 2))
    traits_combinations = list(itertools.combinations(traits, 2))
    for idx, combo in enumerate(combinations):
        similarities = util.cos_sim(combo[0], combo[1])
        mean_similarity = similarities.mean()
        avg_similarity_within_eight_trait += mean_similarity
        count += 1
        key = traits_combinations[idx][0] + " vs " + traits_combinations[idx][1]     
        similarity_within_eight_trait[key] = mean_similarity
    avg_similarity_within_eight_trait /= count # 0.2309
    print("similarity within eight trait: ", similarity_within_eight_trait)
    print(Fore.RED + "average similarity within eight trait: " + str(avg_similarity_within_eight_trait) + Style.RESET_ALL)
     


def main():
    calculate_similarity_between_traits()

if __name__ == "__main__":
    main()


