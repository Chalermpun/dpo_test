import pandas as pd
from datasets import DatasetDict, Dataset
from itertools import combinations
from sklearn.model_selection import train_test_split


def dpo_data_prior(
    df,
    col_ij="rank_of_informative_responses",
    question_col="question",
    context_col="context",
    reference_col="reference",
    background=None,
) -> Dataset:
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
    df_combinations = Dataset.from_pandas(df_combinations)
    df_combinations = df_combinations.remove_columns(original_columns)
    df_combinations = df_combinations.remove_columns("__index_level_0__")
    return df_combinations


def dpo_data(jsonl_path: str, background=True, test_size=0.0001) -> DatasetDict:
    if background:
        prompt_template_real = "### USER:\n{instruction}\n\n### RESPONSE:\n"
    else:
        prompt_template_real = None

    ranking = pd.read_json(jsonl_path, lines=True)
    ranking = dpo_data_prior(
        ranking, col_ij="rank_all", background=prompt_template_real
    )

    train_df = ranking.to_pandas()
    train_df["text_chosen"] = train_df["text_chosen"].apply(lambda x: x[1]["content"])
    train_df["text_rejected"] = train_df["text_rejected"].apply(
        lambda x: x[1]["content"]
    )
    train_df = train_df.dropna()
    train, test = train_test_split(train_df, test_size=test_size)
    dataset = DatasetDict(
        {"train": Dataset.from_pandas(train), "test": Dataset.from_pandas(test)}
    )
    return dataset


# raw_datasets = dpo_data("responses_ranking.jsonl")
# train = raw_datasets['train'].select(range(500))
# test = raw_datasets['test'].select(range(500))
# raw_datasets = DatasetDict({'train':train, 'test': test})
