# adapted from: https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot/scripts/preprocessing.py

import datasets
from datasets import DatasetDict
from typing import Literal
from transformers import AutoTokenizer

chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### USER:\n' + message['content'] }}\n\n{% elif message['role'] == 'assistant' %}\n{{ '### RESPONSE:\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### RESPONSE:' }}\n{% endif %}\n{% endfor %}"

tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/sandbox/model_sealion7b", trust_remote_code=True
)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "right"
tokenizer.chat_template = chat_template


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_hash(example):
    """Get hash of text field."""
    return {"hash": hash(example["text"])}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def filter(example, uniques):
    """Filter dataset with unique values."""
    if not check_uniques(example, uniques):
        return False
    else:
        return True


# dataset = datasets.load_dataset("csv", data_files={"train": "train.csv", "validation": "valid.csv"})
dataset = datasets.load_from_disk("/root/flan_dataset/flan_v2")

dataset = dataset.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "task": "sft",
    },
    num_proc=8,
    desc="Applying chat template",
)

# TRAIN SPLIT DEDUPLICATION

len_train = len(dataset["train"])
print(f"Size of original dataset train: {len_train}")

dataset["train"] = dataset["train"].map(get_hash, num_proc=8, writer_batch_size=10000)

# Deduplicate hashes
uniques = set(dataset["train"].unique("hash"))
frac = len(uniques) / len(dataset["train"])
print(f"Fraction of duplicates: {1-frac:.2%}")

# Deduplicate data
dataset_train_deduplicated = dataset["train"].filter(
    filter, fn_kwargs={"uniques": uniques}
)
print(f"Size of filtered dataset train: {len(dataset_train_deduplicated)}")


# SAVE DEDUPLICATED DATASET
dataset_train_deduplicated = dataset_train_deduplicated.remove_columns(["hash", "text"])

dataset = DatasetDict({"train": dataset_train_deduplicated})
dataset.save_to_disk("/workspace/flan_dataset/flan_v2")
