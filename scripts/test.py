from datasets.arrow_dataset import re
import pandas as pd
from transformers import AutoTokenizer
from alignment import (
    get_datasets,
)
from rich.pretty import pprint


tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/sealion/examples/fine_tuning/models_adepter_7B_V31/",
    trust_remote_code=True,
)

data_args = {"/workspace/flan_dataset/flan": 0.01}
raw_datasets = get_datasets(
    data_args,
    splits=["train"],
)

# raw_datasets = raw_datasets.map(reformat_rawdataset)
# raw_datasets.save_to_disk("/workspace/flan_dataset/flan")


tokenizer = AutoTokenizer.from_pretrained(
    "aisingapore/sea-lion-7b", trust_remote_code=True
)




def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


dataset = raw_datasets.map(
    formatting_prompts_func,
    batched=True,
)
dataset["train"][0]["text"]

# raw_datasets.save_to_disk("flan_2")
template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### USER:\n' + message['content'] }}\n\n{% elif message['role'] == 'assistant' %}\n{{ '### RESPONSE:\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### RESPONSE:' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = template
res = tokenizer.apply_chat_template(raw_datasets['train'][222]['messages'])

tokenizer.decode(res)

print(tokenizer.chat_template)
