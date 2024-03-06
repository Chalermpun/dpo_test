import pandas as pd
from datasets import DatasetDict, Dataset, load_dataset
from itertools import combinations
from sklearn.model_selection import train_test_split


def dpo_data_th(
    df,
    col_ij="rank_of_informative_responses",
    question_col="question",
    context_col="context",
    reference_col="reference",
    select_num=None,
    background=None,
) -> pd.DataFrame:
    def generate_combinations(row):
        responses = row[col_ij]
        question = row[question_col]  # Capture the question for the current row
        context = row[context_col]  # Capture the context for the current row
        reference = row[reference_col]  # Capture the context for the current row
        # Include question and context in the result
        comb = combinations(responses, 2)
        result = [
            (question, context, reference, f"{c[0]['content']}", f"{c[1]['content']}")
            for c in comb
        ]
        return result

    combinations_list = []
    for index, row in df.iterrows():
        combinations_list.extend(generate_combinations(row))

    # Adjust the columns list to include "question" and "context"
    df_combinations = pd.DataFrame(
        combinations_list,
        columns=["question", "context", "reference", "chosen", "rejected"],
    )
    original_columns = df_combinations.columns
    df_combinations = df_combinations.drop_duplicates()
    df_combinations["text_prompt"] = (
        "Context (บริบท): "
        + df_combinations["context"]
        + "\nQuestion (คําถาม): "
        + df_combinations["question"]
    )
    if background:
        df_combinations["text_prompt"] = df_combinations["text_prompt"].apply(
            lambda x: background.format(instruction=x)
        )

    df_combinations["text_chosen"] = df_combinations.apply(
        lambda x: [
            {"content": x["text_prompt"], "role": "user"},
            {"content": x["chosen"], "role": "assistant"},
        ],
        axis=1,
    )
    df_combinations["text_rejected"] = df_combinations.apply(
        lambda x: [
            {"content": x["text_prompt"], "role": "user"},
            {"content": x["rejected"], "role": "assistant"},
        ],
        axis=1,
    )
    if select_num:
        df_combinations = df_combinations.sample(n=select_num)

    df_combinations = df_combinations.drop(original_columns, axis=1)
    return df_combinations


def dpo_data_en(english_data_path: str, select_num: int = None, background=None):

    dataset = load_dataset(
        english_data_path,
        split="train_prefs",
        use_auth_token=True,
    )

    original_columns = dataset.column_names

    def return_prompt_and_responses(samples):
        return {
            "text_prompt": [prompt for prompt in samples["prompt"]],
            "text_chosen": samples["chosen"],
            "text_rejected": samples["rejected"],
        }

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

    train_df = dataset.to_pandas()

    if background:
        train_df["text_prompt"] = train_df["text_prompt"].apply(
            lambda x: background.format(instruction=x)
        )

    if select_num:
        train_df = train_df.sample(n=select_num)
    return train_df


def dpo_data(
    tokenizer,
    jsonl_path_th: str = None,
    english_data_path: str = "HuggingFaceH4/ultrafeedback_binarized",
    background=True,
    test_size=0.0001,
    random_state=42,
    select_num_en=None,
    select_num_th=None,
) -> DatasetDict:
    if background:
        prompt_template_real = "### USER:\n{instruction}\n\n### RESPONSE:\n"
    else:
        prompt_template_real = None

    if jsonl_path_th and english_data_path:
        ranking_th = pd.read_json(jsonl_path_th, lines=True)
        ranking_th = dpo_data_th(
            ranking_th,
            col_ij="rank_all",
            background=prompt_template_real,
            select_num=select_num_th,
        )
        ranking_en = dpo_data_en(
            english_data_path, select_num=select_num_en, background=prompt_template_real
        )
        train_df = pd.concat([ranking_th, ranking_en])
    elif jsonl_path_th:
        ranking_th = pd.read_json(jsonl_path_th, lines=True)
        ranking_th = dpo_data_th(
            ranking_th,
            col_ij="rank_all",
            background=prompt_template_real,
            select_num=select_num_th,
        )
        train_df = ranking_th
    elif english_data_path:
        ranking_en = dpo_data_en(
            english_data_path, select_num=select_num_en, background=prompt_template_real
        )
        train_df = ranking_en
    else:
        raise ValueError("Please provide either jsonl_path_th or english_data_path")

    train_df["text_chosen"] = train_df["text_chosen"].apply(
        lambda x: x[1]["content"] + tokenizer.eos_token
    )
    train_df["text_rejected"] = train_df["text_rejected"].apply(
        lambda x: x[1]["content"] + tokenizer.eos_token
    )
    train_df = train_df.dropna()
    train, test = train_test_split(
        train_df, test_size=test_size, random_state=random_state
    )
    dataset = DatasetDict(
        {"train": Dataset.from_pandas(train), "test": Dataset.from_pandas(test)}
    )
    print("Here ===> ", dataset["train"][0])
    return dataset
