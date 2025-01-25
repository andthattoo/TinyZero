import datasets
import ast
from ast import Assign, Call, Expr, keyword
import os
import re
from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_code_blocks(content):
    """Extract all Python code blocks from content, handling markdown and comments"""
    code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)
    return [cb.strip() for cb in code_blocks if cb.strip()]


def parse_function_calls(code_str):
    """Extract all function calls from a code string"""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []

    calls = []

    for node in ast.walk(tree):
        if isinstance(node, (Assign, Expr)) and isinstance(getattr(node, 'value', None), Call):
            call = node.value
        else:
            continue

        func_name = getattr(call.func, 'id', None)
        if not func_name:
            continue

        kwargs = {}
        for kw in call.keywords:
            arg_name = kw.arg
            try:
                arg_value = ast.literal_eval(kw.value)
                kwargs[arg_name] = arg_value
            except:
                continue

        calls.append((func_name, kwargs))

    return calls


def create_reward_entry(example, idx, split):
    messages = example["messages"]

    # Get last user and assistant messages
    user_messages = [m for m in messages if m["role"] == "user"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]

    if not user_messages or not assistant_messages:
        return None

    user_content = user_messages[-1]["content"]
    assistant_content = assistant_messages[-1]["content"]

    # Extract and parse all code blocks
    code_blocks = extract_code_blocks(assistant_content)
    if not code_blocks:
        return None

    # Use last code block (most recent correction)
    all_calls = parse_function_calls(code_blocks[-1])
    if not all_calls:
        return None

    # Convert to reward model format
    return {
        "data_source": "fc_merged",
        "prompt": [{"role": "user", "content": user_content}],
        "ability": "function_calling",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "expected_calls": [
                    {"function": fn, "arguments": args}
                    for fn, args in all_calls
                ]
            }
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "original_code_blocks": code_blocks
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
            with_indices=True,
            remove_columns=raw_split.column_names
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