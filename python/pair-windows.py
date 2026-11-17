"""Canonical bounded-tail window geometry shared by training and inference.

Native slices preserve offsets and token IDs. Action structure comes only from
canonical fields, never from text delimiters. This module has no model state.
"""
import json
from collections import namedtuple

PAIR_FORMAT = "intent-action-window-bag-v2"
SIGNATURE = ("input_ids", "attention_mask", "token_type_ids")
GEOMETRY = {"max_window_tokens": 323, "max_payload_tokens": 320, "overlap_tokens": 0,
            "max_bag_windows": 1, "max_request_windows": 32,
            "intent_policy": "tail", "description_policy": "head",
            "max_intent_tokens": 96, "max_action_tokens": 96,
            "allocation": "intent-cap-then-description",
            "order": "intent-major-description-minor", "hidden_size": 32,
            "pooled_size": 128, "position_size": 4, "aggregator": "torch-grucell"}
# Pinned cross-encoder bases. Both use the bert-base-uncased vocabulary, so window
# geometry is identical; a learned projection maps each pooled output to pooled_size.
# MiniLM (6 layers, 384 wide) decides on the bi-encoder shortlist.
MODEL_BASES = {"cross-encoder/ms-marco-TinyBERT-L2-v2": "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
               "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a"}
Bag = namedtuple("Bag", "windows positions intent_spans description_spans source_lengths")


class ContextOverflow(ValueError):
    pass


def render_first(intent, action):
    return intent if action is None else intent + "\nAction: " + action["toolName"] + "\n" + action["summary"]


def geometry(a, b, suffix):
    """Fixed payload budget: whole action, at most 96 recent intent tokens, then description."""
    if suffix > GEOMETRY["max_action_tokens"]:
        raise ContextOverflow("action_suffix_budget")
    keep_intent = min(a, GEOMETRY["max_intent_tokens"])
    keep_description = min(b, GEOMETRY["max_payload_tokens"] - suffix - keep_intent)
    return [(a - keep_intent, a)], [(0, keep_description)]


def _truncate_owned(encoding, length, direction):
    """Trim a freshly owned encoding before copying its bounded native fields.

    Native truncation keeps overflow encodings. Rebuild only the small retained
    piece, without serializing or deep-copying the discarded full input.
    """
    from tokenizers import Encoding
    if len(encoding) <= length:
        return encoding
    if length == 0:
        return Encoding()
    encoding.truncate(length, stride=0, direction=direction)
    state = {"ids": encoding.ids, "type_ids": encoding.type_ids, "tokens": encoding.tokens,
             "words": encoding.word_ids, "offsets": encoding.offsets,
             "special_tokens_mask": encoding.special_tokens_mask,
             "attention_mask": encoding.attention_mask, "overflowing": [], "sequence_ranges": {}}
    retained = Encoding()
    retained.__setstate__(json.dumps(state).encode())
    return retained


def fields(encoding):
    return dict(input_ids=encoding.ids, attention_mask=encoding.attention_mask,
                token_type_ids=encoding.type_ids)


def build_bags(tokenizer, intent, action, descriptions):
    """Prepare one rendered context and reuse its bounded tail across candidates."""
    if not descriptions:
        return []
    first = tokenizer.encode(render_first(intent, action), add_special_tokens=False)
    original_length = len(first)
    boundary = original_length
    if action is not None:
        boundary = next((i for i, (_, end) in enumerate(first.offsets) if end > len(intent)), original_length)
    suffix_length = original_length - boundary
    aa, _ = geometry(boundary, 0, suffix_length)
    start, end = aa[0]
    first = _truncate_owned(first, end - start + suffix_length, "left")
    description_capacity = GEOMETRY["max_payload_tokens"] - len(first)
    intent_position = [start / boundary if boundary else 0., end / boundary if boundary else 0.]
    bags = []
    for description in descriptions:
        second = tokenizer.encode(description, add_special_tokens=False)
        description_length = len(second)
        keep = min(description_length, description_capacity)
        second = _truncate_owned(second, keep, "right")
        window = tokenizer.post_process(first, second, add_special_tokens=True)
        positions = [intent_position + [0., keep / description_length if description_length else 0.]]
        bags.append(Bag([fields(window)], positions, [(start, end)], [(0, keep)],
                        (boundary, suffix_length, description_length)))
    return bags


def build_bag(tokenizer, intent, action, description):
    return build_bags(tokenizer, intent, action, (description,))[0]


def admit_request(bags):
    if sum(len(bag.windows) for bag in bags) > GEOMETRY["max_request_windows"]:
        raise ContextOverflow("request_window_budget")
    return bags


def tensor(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": shape}


def graph_contracts():
    return {
        "encoder": {"file": "encoder.onnx", "inputs": [tensor(key, "int64", ["batch", "sequence"]) for key in SIGNATURE],
                    "outputs": [tensor("pooled", "float32", ["batch", 128])]},
        "aggregator": {"file": "aggregator-step.onnx", "inputs": [tensor("pooled", "float32", ["batch", 128]),
            tensor("positions", "float32", ["batch", 4]), tensor("hidden", "float32", ["batch", 32])],
            "outputs": [tensor("hidden_out", "float32", ["batch", 32]), tensor("logit", "float32", ["batch", 1])]}}


def padded_arrays(windows, pad_id, np):
    width = max(len(window["input_ids"]) for window in windows)
    tensors = {}
    for key in SIGNATURE:
        array = np.full((len(windows), width), pad_id if key == "input_ids" else 0, dtype=np.int64)
        for row, window in enumerate(windows):
            array[row, :len(window[key])] = window[key]
        tensors[key] = array
    return tensors


def score_bag(encoder, aggregator, bag, pad_id, np):
    # State is local to this invocation, including failures and cancellation.
    hidden = np.zeros((1, 32), dtype=np.float32)
    for start in range(0, len(bag.windows), 4):
        pooled = encoder.run(["pooled"], padded_arrays(bag.windows[start:start + 4], pad_id, np))[0]
        if pooled.shape != (min(4, len(bag.windows) - start), 128) or not np.isfinite(pooled).all():
            raise ValueError("invalid_encoder_output")
        for offset in range(len(pooled)):
            hidden, logit = aggregator.run(["hidden_out", "logit"], {
                "pooled": pooled[offset:offset + 1], "positions": np.asarray([bag.positions[start + offset]], dtype=np.float32),
                "hidden": hidden})
            if hidden.shape != (1, 32) or logit.shape != (1, 1) or not np.isfinite(hidden).all() or not np.isfinite(logit).all():
                raise ValueError("invalid_aggregator_output")
    return float(logit[0, 0])
