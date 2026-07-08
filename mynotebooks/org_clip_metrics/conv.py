# import csv
import json

# input_csv = "head_spec_clip.csv"
# output_json = "head_spec_clip.json"

# result = {}
# with open(input_csv) as f:
#     for row in csv.DictReader(f):
#         name = row["Head_Name"]
#         result[name] = {k: float(v) for k, v in row.items() if k not in ("Head_Number", "Head_Name")}

# with open(output_json, "w") as f:
#     json.dump(result, f, indent=2)

# print(f"Written {len(result)} entries to {output_json}")


# print(f"Written {len(result)} entries to {output_json}")

# Convert all_group_scores.json: {Head_0: {Head_Name, Scores}} -> {"Layer 8 Head 0": {scores}}
with open("all_group_scores.json") as f:
    raw = json.load(f)

group_scores = {v["Head_Name"]: v["Scores"] for v in raw.values()}

with open("all_group_scores_flat.json", "w") as f:
    json.dump(group_scores, f, indent=2)

print(f"Written {len(group_scores)} entries to all_group_scores_flat.json")