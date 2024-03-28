from templates import PATTERNS
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import json
import glob
import re
import pandas as pd
import numpy.random as random
from typing import List


def scb_translation(example):
    example["en"] = example["translation"]["en"]
    example["th"] = example["translation"]["th"]
    return example


def read_file_xp3x():

    path = "/workspace/sealion/examples/xp3x/content/drive/MyDrive/xp3/*"
    all_file = []
    j = 0
    for file in glob.glob(path):
        j = j + 1
        try:
            with open(file) as f:
                data = [json.loads(line) for line in f]

                all_file.append(file)
        except Exception as e:
            # print(e)
            continue
    print("all file is ", j)
    return all_file


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


def context_bot_human(example):
    pattern = r"<human>:\s*(.*?)\s*<context>:\s*(.*?)\s*<bot>:\s*(.*)"
    extract = re.match(pattern, example["text"])
    human, context, bot = extract.groups()
    example["human"] = human
    example["context"] = context
    example["bot"] = bot
    return example


def thai_investment_consultant_licensing_exams_answer(example):
    def extract_chosen_answer_adjusted(text, choice_number):

        if (
            str(choice_number) + "." in text
            and str(choice_number + 1) + "." not in text
        ):
            pattern = re.compile(rf"{choice_number}\.\s*(.*)")
        else:
            pattern = re.compile(
                rf"{choice_number}\.\s*(.*?)(?=\s{choice_number + 1}\.)"
            )

        pattern2 = re.compile(rf"{choice_number}\.\s*(\d+\.*\d*%)")
        pattern3 = re.compile(rf"{choice_number}\.\s*(\d\.\d\d*\sเท่า)")
        match = pattern.search(text)
        match2 = pattern2.search(text)
        match3 = pattern3.search(text)
        if match:
            extracted_text = match.group(1)
            return str(choice_number) + ". " + extracted_text
        elif match2:
            extracted_text = match2.group(1)
            return str(choice_number) + ". " + extracted_text
        elif match3:
            extracted_text = match3.group(1)
            return str(choice_number) + ". " + extracted_text
        else:
            pattern = rf"{choice_number}\)\s(.*)\s\d\)*\s"
            match = re.search(pattern, text)
            if match:
                return str(choice_number) + ". " + match.group(1).split()[0]
            else:
                raise ValueError(f"Choice not found. {text}\nnumber: {choice_number}")

    example["result"] = extract_chosen_answer_adjusted(
        example["input"], example["result"]
    )
    return example


def wongnai_reviews_sentiment(
    example,
    feelings: List[dict] = [
        {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
        {2: "negative", 1: "neutral", 0: "positive"},
        {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
    ],
):
    def convert_star_rating_to_sentiment(rating):
        if rating == 0:
            return feeling[2]
        elif rating == 1:
            return feeling[2]
        elif rating == 2:
            return feeling[2]
        elif rating == 3:
            return feeling[1]
        elif rating == 4:
            return feeling[0]
        elif rating == 5:
            return feeling[0]
        else:
            raise ValueError(f"Invalid rating: {rating}")

    feeling = random.choice(feelings)
    example["sentiment_th"] = convert_star_rating_to_sentiment(example["star_rating"])
    example["neu"] = feeling[1]
    example["pos"] = feeling[0]
    example["neg"] = feeling[2]
    return example


def thai_sentiment_analysis_dataset_answer(
    example,
    feelings: List[dict] = [
        {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
        {2: "negative", 1: "neutral", 0: "positive"},
        {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
    ],
):
    def answer_sentiment(answer):
        if answer == "neg":
            return feeling[2]
        elif answer == "neu":
            return feeling[1]
        elif answer == "pos":
            return feeling[0]
        else:
            raise ValueError(f"Invalid answer: {answer}")

    feeling = random.choice(feelings)
    example["answer_th"] = answer_sentiment(example["answer"])
    example["neu"] = feeling[1]
    example["pos"] = feeling[0]
    example["neg"] = feeling[2]
    return example


def save_list_to_jsonl(data_list, filename):
    """
    Saves a list of dictionaries to a JSON Lines file.

    Args:
        data_list (list): The list of dictionaries to save.
        filename (str): The name of the file to save the data to.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for item in data_list:
            json_line = json.dumps(item)  # Convert dictionary to JSON string
            file.write(json_line + "\n")


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


def reformat_rawdataset(examples):
    if examples["input"]:
        examples["prompt"] = examples["instruction"] + "\n" + examples["input"]
    else:
        examples["prompt"] = examples["instruction"]
    examples["messages"] = [
        {"content": examples["prompt"], "role": "user"},
        {"content": examples["output"], "role": "assistant"},
    ]
    return examples


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


def create_flan_dataset(split="train"):

    ################# load dataset from huggingface
    xlsum = load_dataset("csebuetnlp/xlsum", "thai", split=split)

    thaisum = load_dataset("thaisum", split=split)

    scb_enth = load_dataset("scb_mt_enth_2020", "enth", split=split)
    scb_enth = scb_enth.map(scb_translation)

    han = load_dataset("pythainlp/han-instruct-dataset-v1.0", split=split)
    han = han.rename_columns({"inputs": "q", "targets": "a"})

    all_file_xp3x = read_file_xp3x()
    xp3x = load_dataset("json", data_files=all_file_xp3x, split=split)
    xp3x = xp3x.filter(lambda example: example["config"] == "eng_Latn-tha_Thai")
    xp3x = DatasetDict({"train": xp3x})

    platypus = load_dataset("garage-bAInd/Open-Platypus", split=split)

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

    thai_food = load_dataset("pythainlp/thai_food_v1.0", split=split)

    thai_wiki_dataset_v3 = load_dataset("pythainlp/thai-wiki-dataset-v3", split=split)

    klongklon = load_dataset("pythainlp/klongklon", split=split)
    klongklon = klongklon.map(context_bot_human)

    thai_investment_consultant_licensing_exams = load_dataset(
        "openthaigpt/thai-investment-consultant-licensing-exams", split=split
    )
    thai_investment_consultant_licensing_exams = (
        thai_investment_consultant_licensing_exams.filter(
            lambda example: example["input"] != None
        )
    )

    thai_investment_consultant_licensing_exams = (
        thai_investment_consultant_licensing_exams.filter(
            lambda example: str(example["result"]) in example["input"]
        )
    )

    thai_investment_consultant_licensing_exams = (
        thai_investment_consultant_licensing_exams.map(
            thai_investment_consultant_licensing_exams_answer
        )
    )

    thai_usembassy = load_dataset("pythainlp/thai_usembassy", split=split)

    wongnai_reviews = load_dataset("wongnai_reviews", split=split)
    wongnai_reviews = wongnai_reviews.map(
        wongnai_reviews_sentiment,
        fn_kwargs={
            "feelings": [
                {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
                {2: "negative", 1: "neutral", 0: "positive"},
                {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
            ]
        },
    )

    tacas61 = pd.read_csv(
        "https://github.com/PyThaiNLP/thai-sentiment-analysis-dataset/raw/master/tcas61.csv",
        sep="\t",
        names=["text", "answer"],
    )
    review_shopping = pd.read_csv(
        "https://github.com/PyThaiNLP/thai-sentiment-analysis-dataset/raw/master/review_shopping.csv",
        sep="\t",
        names=["text", "answer"],
    )
    general_amy = pd.read_csv(
        "https://github.com/PyThaiNLP/thai-sentiment-analysis-dataset/raw/master/general-amy.csv",
        sep="\t",
        names=["text", "answer"],
    )
    thai_sentiment_analysis_dataset = Dataset.from_pandas(
        pd.concat([tacas61, review_shopping, general_amy]).dropna()
    ).remove_columns(["__index_level_0__"])

    thai_sentiment_analysis_dataset = thai_sentiment_analysis_dataset.map(
        thai_sentiment_analysis_dataset_answer,
        fn_kwargs={
            "feelings": [
                {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
                {2: "negative", 1: "neutral", 0: "positive"},
                {2: "รู้สึกไม่ดี", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
            ]
        },
    )

    thai_english_transliteration_dictionary = load_dataset(
        "csv",
        data_files="https://github.com/wannaphong/thai-english-transliteration-dictionary/raw/main/dict.tsv",
        split=split,
        sep="\t",
    )

    prd_news_30112023 = load_dataset("pythainlp/prd_news_30112023", split=split)

    aya_dataset = load_dataset("CohereForAI/aya_collection", "aya_dataset", split=split)
    aya_dataset = aya_dataset.filter(lambda example: example["language"] == "tha")

    aya_collection_templated_xlel_wd = load_dataset(
        "CohereForAI/aya_collection", "templated_xlel_wd", split=split
    )

    aya_collection_templated_xlel_wd = aya_collection_templated_xlel_wd.filter(
        lambda examples: examples["language"] == "tha"
    )

    ################# reformat
    # summarization
    xlsum_list = reformat(xlsum, "xlsum", "csebuetnlp/xlsum", "summarization")
    thaisum_list = reformat(thaisum, "thaisum", "thaisum", "summarization")

    # translation
    scb_enth_list = reformat(
        scb_enth, "scb_mt_en_th", "scb_mt_enth_2020", "translation"
    )
    xp3x_list = reformat(xp3x, "xp3x_enth", "CohereForAI/xP3x", "other")

    han_list = reformat(
        han, "han", "pythainlp/han-instruct-dataset-v1.0", "text-generation"
    )
    platypus_list = reformat(
        platypus, "platypus", "garage-bAInd/Open-Platypus", "other"
    )

    # sentiment_analysis
    wisesight_sentiment_list = reformat(
        wisesight_sentiment,
        "wisesight_sentiment",
        "wisesight_sentiment",
        "text-classification",
    )

    thai_food_list = reformat(
        thai_food, "thai_food", "pythainlp/thai_food_v1.0", "text-generation"
    )

    thai_wiki_dataset_v3_list = reformat(
        thai_wiki_dataset_v3,
        "thai_wiki_dataset_v3",
        "pythainlp/thai-wiki-dataset-v3",
        "question-answering",
    )

    klongklon_list = reformat(
        klongklon, "klongklon", "pythainlp/klongklon", "text-generation"
    )

    thai_investment_consultant_licensing_exams_list = reformat(
        thai_investment_consultant_licensing_exams,
        "thai_investment_consultant_licensing_exams",
        "openthaigpt/thai-investment-consultant-licensing-exams",
        "question-answering",
    )

    thai_usembassy_list = reformat(
        thai_usembassy, "thai_usembassy", "pythainlp/thai_usembassy", "translation"
    )

    wongnai_reviews_list = reformat(
        wongnai_reviews, "wongnai_reviews", "wongnai_reviews", "text-classification"
    )

    thai_sentiment_analysis_dataset_list = reformat(
        thai_sentiment_analysis_dataset,
        "thai_sentiment_analysis_dataset",
        "https://github.com/PyThaiNLP/thai-sentiment-analysis-dataset/",
        "text-classification",
    )

    thai_english_transliteration_dictionary_list = reformat(
        thai_english_transliteration_dictionary,
        "thai_english_transliteration_dictionary",
        "https://github.com/wannaphong/thai-english-transliteration-dictionary/",
        "translation",
    )

    prd_news_30112023_list = reformat(
        prd_news_30112023,
        "prd_news_30112023",
        "pythainlp/prd_news_30112023",
        "text-generation",
    )

    # aya_collection
    aya_dataset_list = reformat(
        aya_dataset,
        "aya_dataset",
        "CohereForAI/aya_collection/aya_dataset",
        "text-generation",
    )

    aya_collection_templated_xlel_wd_list = reformat(
        aya_collection_templated_xlel_wd,
        "aya_collection_templated_xlel_wd",
        "CohereForAI/aya_collection/templated_xlel_wd",
        "text-generation",
    )

    flan_list = (
        xlsum_list
        + thaisum_list
        + scb_enth_list
        + han_list
        + xp3x_list
        + platypus_list
        + wisesight_sentiment_list
        + thai_food_list
        + thai_wiki_dataset_v3_list
        + klongklon_list
        + thai_investment_consultant_licensing_exams_list
        + thai_usembassy_list
        + wongnai_reviews_list
        + thai_sentiment_analysis_dataset_list
        + thai_english_transliteration_dictionary_list
        + prd_news_30112023_list
        + aya_dataset_list
        + aya_collection_templated_xlel_wd_list
    )
    return flan_list


if __name__ == "__main__":

    flan_list = create_flan_dataset()
    flan_dataset = Dataset.from_list(flan_list)
    flan_dataset_dict = DatasetDict({"train": flan_dataset})
    flan_dataset_dict = flan_dataset_dict.map(reformat_rawdataset)
    flan_dataset_dict.save_to_disk("/workspace/flan_dataset/flan")
