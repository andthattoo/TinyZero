import datasets
import ast
from ast import Assign, Call, Expr, keyword
import os
from verl.utils.hdfs_io import copy, makedirs
import argparse


def parse_function_call(code_str):
    """Extract function name and arguments from code string"""
    try:
        code_str = code_str.split("```python")[1].split("```")[0].strip()
        tree = ast.parse(code_str)
    except:
        return None, None

    for node in ast.walk(tree):
        if isinstance(node, Assign) and isinstance(node.value, Call):
            call = node.value
        elif isinstance(node, Expr) and isinstance(node.value, Call):
            call = node.value
        else:
            continue

        func_name = call.func.id
        kwargs = {}
        for kw in call.keywords:
            arg_name = kw.arg
            try:
                arg_value = ast.literal_eval(kw.value)
                kwargs[arg_name] = arg_value
            except:
                continue
        return func_name, kwargs

    return None, None


def create_reward_entry(example, idx, split):
    # Extract dialog components
    messages = example["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")

    # Parse ground truth from assistant response
    func_name, kwargs = parse_function_call(assistant_content)
    if not func_name or not kwargs:
        return None  # Skip malformed entries

    return {
        "data_source": "fc_merged",
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "function_calling",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "function_name": func_name,
                "arguments": kwargs
            }
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "original_assistant_response": assistant_content
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/function_calling')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    dataset = datasets.load_dataset("driaforall/fc_merged")

    def process_split(split):
        raw_split = dataset[split]
        processed = raw_split.map(
            function=lambda ex, idx: create_reward_entry(ex, idx, split),
            with_indices=True
        ).filter(lambda x: x is not None)
        return processed

    train_dataset = process_split("train")
    val_dataset = process_split("validation")
    test_dataset = process_split("test")

    # Save datasets
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(local_dir, args.hdfs_dir)


if __name__ == "__main__":
    main()