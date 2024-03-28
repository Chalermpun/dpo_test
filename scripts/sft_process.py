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
