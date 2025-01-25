import re
import ast
from ast import Assign, Call, Expr
from typing import Dict, List


def extract_function_calls(solution_str: str) -> List[Dict]:
    """Extract function calls from solution code blocks"""
    code_blocks = re.findall(r'```python(.*?)```', solution_str, re.DOTALL)
    if not code_blocks:
        return []

    calls = []
    for code in code_blocks:
        try:
            tree = ast.parse(code.strip())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (Assign, Expr)) and isinstance(getattr(node, 'value', None), Call):
                call = node.value
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

                calls.append({"function": func_name, "arguments": kwargs})

    return calls


def validate_arguments(gen_args: Dict, expected_args: Dict) -> bool:
    """Validate generated arguments match expected types and values"""
    if set(gen_args.keys()) != set(expected_args.keys()):
        return False

    for key, expected_val in expected_args.items():
        gen_val = gen_args.get(key)
        if not isinstance(gen_val, type(expected_val)):
            return False

        # Add type-specific validation if needed
        if isinstance(expected_val, str) and gen_val != expected_val:
            return False
        if isinstance(expected_val, (int, float)) and abs(gen_val - expected_val) > 1e-5:
            return False

    return True


def compute_score(solution_str: str, ground_truth: Dict, partial_credit: float = 0.5) -> float:
    """
    Compute reward score for function calling tasks

    Args:
        solution_str: Generated code with function calls
        ground_truth: Dict with 'expected_calls' list containing
                      {'function': name, 'arguments': dict}
        partial_credit: Score for partially correct calls (0-1)
    """
    expected_calls = ground_truth.get('expected_calls', [])
    if not expected_calls:
        return 0.0

    generated_calls = extract_function_calls(solution_str)
    if not generated_calls:
        return 0.0

    score = 0.0
    matched = set()

    # Match generated calls to expected calls
    for expected in expected_calls:
        for i, gen in enumerate(generated_calls):
            if i in matched:
                continue

            if gen['function'] == expected['function']:
                if gen['arguments'] == expected['arguments']:
                    score += 1.0
                    matched.add(i)
                    break
                elif validate_arguments(gen['arguments'], expected['arguments']):
                    score += partial_credit
                    matched.add(i)
                    break

    # Normalize score by number of expected calls
    max_score = len(expected_calls)
    return min(score / max_score, 1.0) if max_score > 0 else 0.0