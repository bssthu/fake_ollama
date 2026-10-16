"""Carry known token counts without inventing zero for unavailable usage."""


def token_count(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if result < 0 or isinstance(value, float) and value != result:
        return None
    return result


def known_counts(values, mapping):
    return {destination: count for source, destination in mapping
            if (count := token_count(values.get(source))) is not None}


def openai_usage(usage):
    result = known_counts(usage, (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")))
    if len(result) == 2:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result
