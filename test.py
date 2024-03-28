from templates import PATTERNS
from typing import List
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
import numpy.random as random
from collections import Counter


def wisesight_sentiment_category(
    example,
    feelings: List[dict] = [
        {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
        {2: "negative", 1: "neutral", 0: "positive"},
        {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
    ],
):
    feeling = random.choice(feelings)
    example["answer"] = feeling[example["category"]]
    example["neu"] = feeling[1]
    example["pos"] = feeling[0]
    example["neg"] = feeling[2]
    return example


def create_wisesignet_sentiment():
    split = "train"
    wisesight_sentiment = load_dataset("wisesight_sentiment", split=split)
    wisesight_sentiment = wisesight_sentiment.filter(
        lambda x: x["category"] in [0, 1, 2]
    )
    wisesight_sentiment = wisesight_sentiment.map(
        wisesight_sentiment_category,
        fn_kwargs={
            "feelings": [
                {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
                {2: "negative", 1: "neutral", 0: "positive"},
                {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
            ]
        },
    )
    wisesight_sentiment = wisesight_sentiment.rename_columns({"texts": "text"})
    return wisesight_sentiment


def reformat(dataset_dict: DatasetDict, pattern_name: str, source: str, task: str):
    """
    Reformats datasets based on given pattern and converts them into a Dataset.
    """
    datas = []  # This list will hold the generated data
    if isinstance(dataset_dict, Dataset):
        generator = dataset_to_generator(dataset_dict, pattern_name, source, task)
        datas.extend(generator)
    elif isinstance(dataset_dict, DatasetDict):
        for key in ["train", "validation", "test"]:
            dataset = dataset_dict[key] if key in dataset_dict else None
            if dataset is not None:
                generator = dataset_to_generator(dataset, pattern_name, source, task)
                datas.extend(generator)  # Extend the list with generated items
    else:
        raise TypeError("dataset_dict must be a Dataset or a DatasetDict")
    return datas


def dataset_to_generator(dataset, pattern_name: str, source: str, task: str):
    if dataset is None:
        return  # If dataset is None, do nothing.
    patterns = PATTERNS[pattern_name]
    patterns = random.choice(patterns, size=len(dataset), replace=True)
    for i, row in enumerate(tqdm(dataset, total=len(dataset))):
        pattern = patterns[i]
        data = {
            "instruction": pattern["instruction"].format(**row),
            "input": pattern["input"].format(**row),
            "output": pattern["output"].format(**row),
            "template": pattern,
            "source": source,
            "task": task,
        }
        yield data
