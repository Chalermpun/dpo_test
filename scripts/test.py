import pandas as pd
from dpo_process import dpo_data_th, dpo_data_en, dpo_data
from transformers import AutoTokenizer

ranking_th = pd.read_json("responses_ranking.jsonl", lines=True)
train_df_th = dpo_data_th(ranking_th, select_num=50)
train_df_en = dpo_data_en("HuggingFaceH4/ultrafeedback_binarized", select_num=50)


tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/sealion/examples/fine_tuning/models_adepter_7B_V31/",
    trust_remote_code=True,
)
raw_datasets = dpo_data(
    tokenizer,
    # jsonl_path_th="responses_ranking.jsonl",
    jsonl_path_th=None,
    english_data_path="HuggingFaceH4/ultrafeedback_binarized",
    select_num_th=50,
    select_num_en=None,
)
