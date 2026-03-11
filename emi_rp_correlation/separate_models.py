import os
import shutil

src_dir = "data/data"
dst_v15 = os.path.join('data', "llava-v1.5-13b")
dst_v16 = os.path.join('data', "llava-v1.6-vicuna-13b")

os.makedirs(dst_v15, exist_ok=True)
os.makedirs(dst_v16, exist_ok=True)

count_v15 = 0
count_v16 = 0

for filename in os.listdir(src_dir):
    filepath = os.path.join(src_dir, filename)
    if not os.path.isfile(filepath) or not filename.endswith(".jsonl"):
        continue

    if filename.endswith("llava-v1.5-13b.jsonl"):
        new_name = filename.replace("_llava-v1.5-13b.jsonl", ".jsonl")
        shutil.copy2(filepath, os.path.join(dst_v15, new_name))
        count_v15 += 1
    elif filename.endswith("llava-v1.6-vicuna-13b.jsonl"):
        new_name = filename.replace("_llava-v1.6-vicuna-13b.jsonl", ".jsonl")
        shutil.copy2(filepath, os.path.join(dst_v16, new_name))
        count_v16 += 1

print(f"Copied {count_v15} files to {dst_v15}")
print(f"Copied {count_v16} files to {dst_v16}")
