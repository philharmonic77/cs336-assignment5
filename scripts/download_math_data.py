from datasets import load_dataset, concatenate_datasets
import json
import re
from pathlib import Path

subjects = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]

all_train = []
all_test = []

for s in subjects:
    d = load_dataset("EleutherAI/hendrycks_math", s)
    all_train.append(d["train"])
    all_test.append(d["test"])

train_dataset = concatenate_datasets(all_train)
test_dataset = concatenate_datasets(all_test)


def extract_answer(solution):
    start = solution.find("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(solution) and depth > 0:
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out) if out else None


def save_jsonl(dataset, path):
    with open(path, "w") as f:
        for row in dataset:
            answer = extract_answer(row["solution"])

            item = {
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer,
                "subject": row["type"],
                "level": row["level"],
            }

            f.write(json.dumps(item) + "\n")


Path("data/math").mkdir(parents=True, exist_ok=True)

save_jsonl(train_dataset, "data/math/train.jsonl")
save_jsonl(test_dataset, "data/math/validation.jsonl")

print("Saved train/test JSONL.")
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
