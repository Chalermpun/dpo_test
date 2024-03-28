from datasets.arrow_dataset import re
import pandas as pd
from transformers import AutoTokenizer
from alignment import (
    get_datasets,
)
from rich.pretty import pprint
from sft_process import reformat_rawdataset


tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/sealion/examples/fine_tuning/models_adepter_7B_V31/",
    trust_remote_code=True,
)

data_args = {"/workspace/flan_dataset/flan": 1}
raw_datasets = get_datasets(
    data_args,
    splits=["train"],
)

# raw_datasets = raw_datasets.map(reformat_rawdataset)
# raw_datasets.save_to_disk("/workspace/flan_dataset/flan")
