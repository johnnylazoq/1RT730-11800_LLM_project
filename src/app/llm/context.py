"""Conservative request budgeting for local models without a tokenizer endpoint."""
import json
import os


class ContextBudgetExceeded(ValueError):
    pass


def context_size(llm=None):
    return int(getattr(llm, 'num_ctx', None) or os.getenv('OLLAMA_NUM_CTX', '16384'))


def request_size(messages, schema):
    """Byte-based token upper estimate for UTF-8/BPE, plus chat-template allowance.

    This intentionally overestimates ordinary English text; it is not an exact
    model token count. Include the output schema because it also costs context.
    """
    schema_data = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else schema
    return (sum(len(str(m.content).encode('utf-8')) + 64 for m in messages)
            + len(json.dumps(schema_data, ensure_ascii=False).encode('utf-8')) + 512)


def check_context(messages, schema, llm=None):
    size = context_size(llm)
    reserve = min(2048, size // 4)
    estimated = request_size(messages, schema)
    if estimated > size - reserve:
        raise ContextBudgetExceeded(
            f'Request budget exceeded: conservative estimate {estimated}, input budget {size - reserve}; '
            f'{reserve} tokens reserved for response. Reduce evidence or increase OLLAMA_NUM_CTX.')
    return estimated
