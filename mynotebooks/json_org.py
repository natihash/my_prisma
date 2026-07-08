# import json
# import re
# import numpy as np

# # --- CONFIGURATION ---
# FILE_PATHS = {
#     "LP": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/new_task_rel_lp.json",
#     "LoRA16": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/new_task_rel_lora16.json",
#     "LoRA4": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/new_task_rel_lora4.json",
#     "FFT": "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/new_task_rel_fft.json"
# }
# OUTPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/for_cld/combined_heads_analysis.json"
# # ---------------------

# def natural_sort_key(head_string):
#     """
#     Extracts numbers from the head string to sort them numerically.
#     Ensures 'Layer 2' comes before 'Layer 11', and 'Head 2' before 'Head 11'.
#     """
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', head_string)]

# # Step 1: Load data and calculate the averages for each file
# file_data = {}
# file_averages = {}

# for file_key, file_path in FILE_PATHS.items():
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             file_data[file_key] = data
            
#             # Extract all values to compute the mean for this specific file
#             rank_means = [metrics["rank_mean"] for metrics in data.values() if "rank_mean" in metrics]
#             top_k_means = [metrics["top_k_mean"] for metrics in data.values() if "top_k_mean" in metrics]
            
#             # Store the baseline averages for distance calculation
#             file_averages[file_key] = {
#                 "avg_rank_mean": np.mean(rank_means) if rank_means else 0.0,
#                 "avg_top_k_mean": np.mean(top_k_means) if top_k_means else 0.0
#             }
#     except FileNotFoundError:
#         print(f"Warning: {file_path} not found. Skipping {file_key}.")
#         continue

# # Step 2: Gather all unique head names across all loaded files
# all_heads = set()
# for data in file_data.values():
#     all_heads.update(data.keys())

# # Sort heads naturally so Layer 0, Layer 1... Layer 11 line up perfectly
# sorted_heads = sorted(list(all_heads), key=natural_sort_key)

# # Step 3: Restructure the data by Attention Head using the sorted order
# output_structure = {}

# for head in sorted_heads:
#     output_structure[head] = {}
    
#     # Keeps the file keys in the exact order: LP, LoRA16, LoRA4, FFT
#     for file_key in FILE_PATHS.keys():
#         if file_key not in file_data or head not in file_data[file_key]:
#             continue
            
#         head_metrics = file_data[file_key][head]
        
#         # Grab individual scores
#         rank_mean = head_metrics.get("rank_mean")
#         top_k_mean = head_metrics.get("top_k_mean")
        
#         # Calculate distances from the file's average (Score - Average)
#         rank_mean_distance = rank_mean - file_averages[file_key]["avg_rank_mean"]
#         top_k_mean_distance = top_k_mean - file_averages[file_key]["avg_top_k_mean"]
        
#         # Save to the new structure
#         output_structure[head][file_key] = {
#             "rank_mean": rank_mean,
#             "rank_mean_distance_from_avg": rank_mean_distance,
#             "top_k_mean": top_k_mean,
#             "top_k_mean_distance_from_avg": top_k_mean_distance
#         }

# # Step 4: Save the consolidated data to the target output path
# import os
# os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # Creates the 'for_cld' directory if it doesn't exist

# with open(OUTPUT_FILE, 'w') as f:
#     json.dump(output_structure, f, indent=2)

# print(f"Success! Combined JSON data saved to:\n{OUTPUT_FILE}")


import json
import re
import os

# --- CONFIGURATION ---
SCORES_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/all_group_scores_imagenet_lp.json"  # Replace with actual path
# SCORES_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/all_group_scores.json"
TEXTS_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/head_top_texts_imagenet_lp.json"    # Replace with actual path
# TEXTS_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/head_top_texts.json"
OUTPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/for_cld/group_text_lp.json"
# ---------------------

def natural_sort_key(head_string):
    """Sorts strings containing numbers logically (Layer 2 before Layer 11)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', head_string)]

# Step 1: Load both JSON files
with open(SCORES_FILE, 'r') as f:
    scores_data = json.load(f)

with open(TEXTS_FILE, 'r') as f:
    texts_data = json.load(f)

# Step 2: Map structural keys ("Head_0", etc.) to human-readable names ("Layer 8 Head 0")
# This ensures data aligns correctly even if keys are ordered differently across files.
scores_by_name = {}
for meta in scores_data.values():
    if "Head_Name" in meta and "Scores" in meta:
        scores_by_name[meta["Head_Name"]] = meta["Scores"]

texts_by_name = {}
for meta in texts_data.values():
    if "Head_Name" in meta and "Top_Texts" in meta:
        texts_by_name[meta["Head_Name"]] = meta["Top_Texts"]

# Step 3: Get a naturally sorted list of all unique head names
all_head_names = sorted(list(set(scores_by_name.keys()).union(texts_by_name.keys())), key=natural_sort_key)

# Step 4: Process and extract top features
output_structure = {}

for head_name in all_head_names:
    output_structure[head_name] = {
        "top_groups": {},
        "top_texts": {}
    }
    
    # Process Group Scores -> Sort by score value descending -> Take Top 6
    if head_name in scores_by_name:
        group_scores = scores_by_name[head_name]
        sorted_groups = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
        top_6_groups = sorted_groups[:6]
        output_structure[head_name]["top_groups"] = dict(top_6_groups)
        
    # Process Text Scores -> Sort by list length descending -> Take Top 20
    if head_name in texts_by_name:
        text_lists = texts_by_name[head_name]
        # Sort based on the length of the list: len(item[1])
        sorted_texts = sorted(text_lists.items(), key=lambda item: len(item[1]), reverse=True)
        # top_20_texts = sorted_texts[:20]
        # output_structure[head_name]["top_texts"] = dict(top_20_texts)
        # Extract only the keys (text labels) from the top 20 items
        top_20_texts = [item[0] for item in sorted_texts[:20]]
        output_structure[head_name]["top_texts"] = top_20_texts

# Step 5: Save the structured JSON output
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_structure, f, indent=2)

print(f"Success! Consolidated JSON file saved to:\n{OUTPUT_FILE}")


# import json
# import os

# # --- CONFIGURATION ---
# # The path to your consolidated JSON file
# INPUT_FILE = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/for_cld/combined_heads_analysis.json"

# # The directory where you want to save the 4 individual JSON files
# OUTPUT_DIR = "/home/nfm/Desktop/rhome/nfm/ViT-Prisma/mynotebooks/org_clip_metrics_real/imagenet1k/for_cld/"
# # ---------------------

# # Step 1: Load the consolidated file
# with open(INPUT_FILE, 'r') as f:
#     combined_data = json.load(f)

# # Step 2: Initialize containers for the four configurations
# # We will dynamically pull keys like "LP", "LoRA16", etc.
# separated_data = {}

# # Step 3: Iterate through every head and split the configurations
# for head_name, configs in combined_data.items():
#     for config_key, metrics in configs.items():
#         # Initialize the config dictionary if it hasn't been seen yet
#         if config_key not in separated_data:
#             separated_data[config_key] = {}
        
#         # Assign the metrics under the current head name for this specific file
#         separated_data[config_key][head_name] = metrics

# # Step 4: Write each configuration to its own separate JSON file
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for config_key, head_data in separated_data.items():
#     # Build filename (e.g., "/.../imagenet1k/LP.json")
#     output_filename = os.path.join(OUTPUT_DIR, f"{config_key}.json")
    
#     with open(output_filename, 'w') as f:
#         json.dump(head_data, f, indent=2)
    
#     print(f"Saved: {output_filename}")

# print("\nSuccessfully split and saved all 4 JSON files!")