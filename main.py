from templates import PATTERNS
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from tqdm import tqdm
import json
import glob
import re
import pandas as pd
import numpy.random as random
import random as random_normal
from typing import List
import numpy as np
from tabulate import tabulate
import os

pd.set_option("display.float_format", None)


def deterministic_random(seed: int) -> random_normal.Random:
    return random_normal.Random(seed)


def load_jsonl_datasets(directory):
    jsonl_files = [file for file in os.listdir(directory) if file.endswith(".jsonl")]
    data_files = [os.path.join(directory, file) for file in jsonl_files]
    print(data_files)
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def scb_translation(example):
    example["en"] = example["translation"]["en"]
    example["th"] = example["translation"]["th"]
    return example


def read_file_xp3x():

    path = "/workspace/sealion_old/examples/xp3x/content/drive/MyDrive/xp3/*"
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
        {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
        {2: "bad", 1: "neutral", 0: "good"},
        {2: "terrible", 1: "neutral", 0: "great"},
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
        {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
        {2: "bad", 1: "neutral", 0: "good"},
        {2: "terrible", 1: "neutral", 0: "great"},
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
        {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
        {2: "bad", 1: "neutral", 0: "good"},
        {2: "terrible", 1: "neutral", 0: "great"},
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


def deEmojify(text):
    regrex_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\ufe0f"
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def dataset_to_generator(dataset, pattern_name: str, source: str, task: str):
    if dataset is None:
        return  # If dataset is None, do nothing.
    patterns = PATTERNS[pattern_name]
    patterns = random.choice(patterns, size=len(dataset), replace=True)
    for i, row in enumerate(tqdm(dataset, total=len(dataset))):
        pattern = patterns[i]
        data = {
            "instruction": deEmojify(pattern["instruction"].format(**row)),
            "input": deEmojify(pattern["input"].format(**row)),
            "output": deEmojify(pattern["output"].format(**row)),
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


def wiki_lingua_mep(example):
    lang_th = ["ไทย", "Thai"]
    lang_en = ["อังกฤษ", "English"]

    lang_th_rand = random.choice(lang_th)
    lang_en_rand = random.choice(lang_en)

    example["source_lang"] = (
        lang_en_rand if example["source_language"] == "en" else lang_th_rand
    )
    example["target_lang"] = (
        lang_en_rand if example["target_language"] == "en" else lang_th_rand
    )
    return example


def add_prefix(example):
    a = example["text"].split("<bot>:")

    example["bot_morethan_one"] = len(a)
    example["has_context"] = 1 if "<context>:" in example["text"] else 0

    v = example["text"]

    # Find the indices of the tags
    context_index = v.find("<context>:")
    human_index = v.find("<human>:")
    bot_index = v.find("<bot>:")

    context = v[context_index:human_index].replace("<context>:", "").strip()
    human = v[human_index:bot_index].replace("<human>:", "").strip()
    bot = v[bot_index:].replace("<bot>:", "").strip()

    combined = ""
    if context != "":
        combined = context + "\n" + human
    else:
        combined = human

    example["Context"] = ""
    example["Instruction"] = combined.strip()
    example["Answer"] = bot.strip()

    return example


def create_flan_dataset(split="train"):

    ### add new dataset by Beer ######

    cache_dir = "/workspace/flan_dataset/cache"

    math_50k = pd.read_json(
        "https://github.com/AGI-Edgerunners/LLM-Adapters/raw/main/ft-training_set/math_50k.json"
    )
    math_50k = Dataset.from_pandas(math_50k)
    commonsense_170k = pd.read_json(
        "https://github.com/AGI-Edgerunners/LLM-Adapters/raw/main/ft-training_set/commonsense_170k.json"
    )
    commonsense_170k = Dataset.from_pandas(commonsense_170k)

    math_50k_list = reformat(math_50k, "math_50k", "math_50k", "text-generation")

    print(np.random.choice(math_50k_list, size=3, replace=False))
    commonsense_170k_list = reformat(
        commonsense_170k, "commonsense_170k", "commonsense_170k", "text-generation"
    )

    print(np.random.choice(commonsense_170k_list, size=3, replace=False))
    dataset_wangchanglm = load_dataset(
        "pythainlp/final_training_set_v1", split=split, cache_dir=cache_dir
    )
    dataset_wangchanglm = dataset_wangchanglm.map(
        add_prefix, load_from_cache_file=False
    )
    dataset_wangchanglm = dataset_wangchanglm.filter(
        lambda x: x["bot_morethan_one"] == 2
    )
    dataset_wangchanglm_list = reformat(
        dataset_wangchanglm, "dataset_wangchanglm", "dataset_wangchanglm", "generation"
    )
    print(np.random.choice(dataset_wangchanglm_list, size=3, replace=False))
    dataset_tiny = load_dataset(
        "nampdn-ai/tiny-codes", split=split, cache_dir=cache_dir
    ).filter(
        lambda example: example["programming_language"]
        in ["JavaScript", "Python", "relation database and SQL"]
    )
    print(dataset_tiny)
    dataset_tiny_list = reformat(
        dataset_tiny, "tiny_code", "dataset_tiny", "generation"
    )
    print(np.random.choice(dataset_tiny_list, size=3, replace=False))

    data_dir = "/workspace/sealion_old/examples/flan_v2/downloads/flan_v2_arrow"
    flan_v2 = load_from_disk(data_dir)
    flan = flan_v2["train"].filter(lambda x: x["task"] == "flan", num_proc=8)
    cot = flan_v2["train"].filter(lambda x: x["task"] == "cot", num_proc=8)
    random_flan_ids = deterministic_random(42).sample(range(len(flan)), k=100000)
    flan = flan.select(random_flan_ids)
    cot_list = reformat(cot, "flan_v2", "cot_v2", "generation")
    flan_list = reformat(flan, "flan_v2", "flan_v2", "generation")

    # flan_v2_dialog = load_dataset("SirNeural/flan_v2", split="train")
    # flan_v2_dialog = flan_v2_dialog.filter(lambda example: example["task"] == "dialog")
    # random_test_ids = random.sample(range(len(flan_v2_dialog)), k=100000)
    # flan_v2_dialog = flan_v2_dialog.select(random_test_ids)
    # print(flan_v2_dialog)
    # flan_v2_dialog_list = reformat(
    #     flan_v2_dialog, "flan_v2", "flan_v2_dialog", "generation"
    # )

    alt_dataset = load_dataset(
        "alt", "alt-parallel", split=split, cache_dir=cache_dir
    )  # .select(range(number_rand))
    alt_dataset = alt_dataset.map(scb_translation, load_from_cache_file=False)
    alt_dataset_list = reformat(
        alt_dataset, "scb_mt_en_th", "alt_dataset", "translation"
    )

    ted_talks_iwslt2014 = load_dataset(
        "ted_talks_iwslt",
        language_pair=("en", "th"),
        year="2014",
        split=split,
        cache_dir=cache_dir,
    )  # .select(range(number_rand))
    ted_talks_iwslt2015 = load_dataset(
        "ted_talks_iwslt",
        language_pair=("en", "th"),
        year="2015",
        split=split,
        cache_dir=cache_dir,
    )  # .select(range(number_rand))
    ted_talks_iwslt2016 = load_dataset(
        "ted_talks_iwslt",
        language_pair=("en", "th"),
        year="2016",
        split=split,
        cache_dir=cache_dir,
    )  # .select(range(number_rand))

    ted_talks_iwslt2014 = ted_talks_iwslt2014.map(
        scb_translation, load_from_cache_file=False
    )
    ted_talks_iwslt2014_list = reformat(
        ted_talks_iwslt2014, "scb_mt_en_th", "ted_talks_iwslt2014", "translation"
    )
    ted_talks_iwslt2015 = ted_talks_iwslt2015.map(
        scb_translation, load_from_cache_file=False
    )
    ted_talks_iwslt2015_list = reformat(
        ted_talks_iwslt2015, "scb_mt_en_th", "ted_talks_iwslt2015", "translation"
    )
    ted_talks_iwslt2016 = ted_talks_iwslt2016.map(
        scb_translation, load_from_cache_file=False
    )
    ted_talks_iwslt2016_list = reformat(
        ted_talks_iwslt2016, "scb_mt_en_th", "ted_talks_iwslt2016", "translation"
    )
    wiki_lingua = load_dataset(
        "GEM/wiki_lingua", "en_th", split=split, cache_dir=cache_dir
    )  # .select(range(number_rand))
    wiki_lingua = wiki_lingua.map(wiki_lingua_mep, load_from_cache_file=False)
    wiki_lingua_list = reformat(
        wiki_lingua, "wiki_lingua", "wiki_lingua", "summarization"
    )
    ### End add new dataset by Beer ######

    ################# load dataset from huggingface
    xlsum = load_dataset("csebuetnlp/xlsum", "thai", split=split, cache_dir=cache_dir)

    thaisum = load_dataset("thaisum", split=split, cache_dir=cache_dir)

    scb_enth = load_dataset(
        "scb_mt_enth_2020", "enth", split=split, cache_dir=cache_dir
    )
    scb_enth = scb_enth.map(scb_translation, load_from_cache_file=False)

    han = pd.read_excel("/workspace/sealion_old/examples/han_instruct-v2.xls")
    han = Dataset.from_pandas(han)
    han_dataset = DatasetDict()
    han_dataset["train"] = han
    han_dataset["train"] = han_dataset["train"].remove_columns("Unnamed: 0")

    all_file_xp3x = read_file_xp3x()
    xp3x = load_dataset("json", data_files=all_file_xp3x, split=split)
    xp3x = xp3x.filter(lambda example: example["config"] in ["thai", "en_th"])
    xp3x = DatasetDict({"train": xp3x})

    platypus = load_dataset(
        "garage-bAInd/Open-Platypus", split=split, cache_dir=cache_dir
    )
    platypus = platypus.filter(
        lambda example: example["data_source"] != "scienceqa"
        and example["data_source"] != "reclor"
    )

    wisesight_sentiment = load_dataset(
        "wisesight_sentiment", split=split, cache_dir=cache_dir
    )
    wisesight_sentiment = wisesight_sentiment.filter(
        lambda x: x["category"] in [0, 1, 2]
    )
    wisesight_sentiment = wisesight_sentiment.map(
        wisesight_sentiment_category,
        fn_kwargs={
            "feelings": [
                {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
                {2: "negative", 1: "neutral", 0: "positive"},
                {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
                {2: "bad", 1: "neutral", 0: "good"},
                {2: "terrible", 1: "neutral", 0: "great"},
            ]
        },
        load_from_cache_file=False,
    )
    wisesight_sentiment = wisesight_sentiment.rename_columns({"texts": "text"})

    thai_food = load_dataset(
        "pythainlp/thai_food_v1.0", split=split, cache_dir=cache_dir
    )

    thai_wiki_dataset_v3 = load_dataset(
        "pythainlp/thai-wiki-dataset-v3", split=split, cache_dir=cache_dir
    )

    klongklon = load_dataset("pythainlp/klongklon", split=split, cache_dir=cache_dir)
    klongklon = klongklon.map(context_bot_human, load_from_cache_file=False)

    thai_investment_consultant_licensing_exams = load_dataset(
        "openthaigpt/thai-investment-consultant-licensing-exams",
        split=split,
        cache_dir=cache_dir,
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

    # thai_investment_consultant_licensing_exams = (
    #     thai_investment_consultant_licensing_exams.map(
    #         thai_investment_consultant_licensing_exams_answer,
    #         load_from_cache_file=False,
    #     )
    # )

    thai_usembassy = load_dataset(
        "pythainlp/thai_usembassy", split=split, cache_dir=cache_dir
    )

    wongnai_reviews = load_dataset("wongnai_reviews", split=split, cache_dir=cache_dir)
    wongnai_reviews = wongnai_reviews.map(
        wongnai_reviews_sentiment,
        fn_kwargs={
            "feelings": [
                {2: "เชิงลบ", 1: "เป็นกลาง", 0: "เชิงบวก"},
                {2: "negative", 1: "neutral", 0: "positive"},
                {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
                {2: "bad", 1: "neutral", 0: "good"},
                {2: "terrible", 1: "neutral", 0: "great"},
            ]
        },
        load_from_cache_file=False,
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
                {2: "รู้สึกแย่", 1: "รู้สึกเฉยๆ", 0: "รู้สึกดี"},
                {2: "bad", 1: "neutral", 0: "good"},
                {2: "terrible", 1: "neutral", 0: "great"},
            ]
        },
        load_from_cache_file=False,
    )

    thai_english_transliteration_dictionary = load_dataset(
        "csv",
        data_files="https://github.com/wannaphong/thai-english-transliteration-dictionary/raw/main/dict.tsv",
        split=split,
        sep="\t",
        cache_dir=cache_dir,
    )

    prd_news_30112023 = load_dataset(
        "pythainlp/prd_news_30112023", split=split, cache_dir=cache_dir
    )

    aya_dataset = load_dataset(
        "CohereForAI/aya_collection", "aya_dataset", split=split, cache_dir=cache_dir
    )
    aya_dataset = aya_dataset.filter(lambda example: example["language"] == "tha")

    aya_collection_templated_xlel_wd = load_dataset(
        "CohereForAI/aya_collection",
        "templated_xlel_wd",
        split=split,
        cache_dir=cache_dir,
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

    han_list = reformat(han_dataset, "han", "han-dataset", "text-generation")
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

    # thai_investment_consultant_licensing_exams_list = reformat(
    #     thai_investment_consultant_licensing_exams,
    #     "thai_investment_consultant_licensing_exams",
    #     "openthaigpt/thai-investment-consultant-licensing-exams",
    #     "question-answering",
    # )

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
        # + thai_investment_consultant_licensing_exams_list
        + thai_usembassy_list
        + wongnai_reviews_list
        + thai_sentiment_analysis_dataset_list
        + thai_english_transliteration_dictionary_list
        + prd_news_30112023_list
        + aya_dataset_list
        + aya_collection_templated_xlel_wd_list
        + alt_dataset_list
        + ted_talks_iwslt2014_list
        + ted_talks_iwslt2015_list
        + ted_talks_iwslt2016_list
        + wiki_lingua_list
        + dataset_wangchanglm_list
        + dataset_tiny_list
        + flan_list
        + cot_list
        + math_50k_list
        + commonsense_170k_list
    )
    return flan_list


def add_new_dataset(dataset1, dataset2, splits=["train", "train_sft"]):
    dataset1 = dataset1[splits[0]].select_columns(["messages", "prompt"])
    dataset2 = dataset2[splits[1]].select_columns(["messages", "prompt"])
    new_dataset = concatenate_datasets([dataset1, dataset2])
    dataset = DatasetDict({"train": new_dataset})
    return dataset


def update_metadata(metadata1, metadata2):
    metadata1.update(metadata2)
    return metadata1


def create_metadata(dataset, split="train", names="ultrachat_200k"):
    dataset = dataset[split].to_pandas()
    if "source" not in dataset.columns:
        metadata = dict()
        metadata[names] = len(dataset)
        return metadata
    else:
        grouped = dataset.groupby(["source"])
        metadata = dict()
        for source, group in grouped:
            metadata[source] = len(group)
        return metadata


def save_metadata(metadata, path):
    metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
    metadata_df = metadata_df.reset_index()
    metadata_df.columns = ["name", "rows"]
    total = metadata_df["rows"].sum()
    metadata_df["name"] = metadata_df["name"].apply(
        lambda x: x if isinstance(x, str) else x[0]
    )
    max_string = metadata_df["name"].str.len().max()
    half_string_max = max_string // 2
    half_string_max = "=" * half_string_max
    string_total = half_string_max + " total " + half_string_max
    metadata_df = metadata_df._append(
        {"name": string_total, "rows": total}, ignore_index=True
    )
    print(tabulate(metadata_df, headers="keys", tablefmt="psql"))
    with open(path, "w") as f:
        f.write(tabulate(metadata_df, headers="keys", tablefmt="psql"))


if __name__ == "__main__":

    flan_list = create_flan_dataset()
    flan_dataset = Dataset.from_list(flan_list)
    flan_dataset_dict = DatasetDict({"train": flan_dataset})
    flan_dataset_dict = flan_dataset_dict.map(reformat_rawdataset)
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")

    metadata_ultrachat = create_metadata(ultrachat, "train_sft")
    metadata_flan = create_metadata(flan_dataset_dict, "train")
    metadata = update_metadata(metadata_ultrachat, metadata_flan)

    raw_datasets = add_new_dataset(flan_dataset_dict, ultrachat)
    raw_datasets.save_to_disk("/root/flan_dataset/flan_v2")
    save_metadata(metadata, "./metadata/metadata_v2.txt")
