# Inference interface

OpenJev returns decisions and probabilities directly. It does not generate a chain-of-thought or explanation text.

## Request format

A request contains one `state` and a nonempty `questions` object. Question IDs associate outputs with inputs. Each question has `type`, `instructions`, and type-specific `criteria`. State, instructions and criteria content can be text or JSON objects/arrays, which are serialized deterministically.

| Type | Input criteria | Output |
| --- | --- | --- |
| `choice` | Object mapping candidate names to descriptions; a description can be null | Selected name, full probabilities, confidence |
| `noul` | Optional object with `false` and `true` descriptions | Probability of true in `noul` |
| `score` | Ordered list of rubric level descriptions | Probabilities over level indices, expected level index, legend, confidence |

Choice supports 1–255 candidates in the interface, within the total token budget. Score supports 1–10 ordered rubric levels. Its expected score uses zero-based indices; use the returned probability vector to compute a different numeric scale if needed. Candidate names are not fixed classes learned by the output head.

The compact dataset's `options` list is converted to `choice.criteria` by using each option's **name** as the key and its optional **criteria** as the value. Names must be unique for that wire format. Dataset rows with duplicate names or custom option IDs can be evaluated through the native `Decision`/serializer/model interface without conflating names or IDs. Score dataset rows likewise preserve their actual names and criteria during training; the wire-format score list is a convenience rubric interface, not a lossless serialization of every dataset row.

Example with all three question types:

```python
request = {
    "state": "A duplicate charge appeared today. The customer recognizes the merchant.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Select the appropriate support team.",
            "criteria": {
                "Duplicate payment": "The same purchase was charged more than once.",
                "Unrecognized payment": "The customer does not recognize the purchase."
            }
        },
        "recognized": {
            "type": "noul",
            "instructions": "Does the customer recognize the merchant?"
        },
        "urgency": {
            "type": "score",
            "instructions": "Assess how urgently this case needs action.",
            "criteria": ["Routine", "Prompt attention", "Immediate action"]
        }
    }
}
```

## Python

```python
from openjev.runtime.engine import DecisionEngine

engine = DecisionEngine("shenjunhao/OpenJev-4B", execution_mode="expanded")
result = engine.predict(request)
print(result["answers"])
```

A local artifact directory can replace the Hub ID. Hub authentication uses the normal `hf auth login` or `HF_TOKEN` mechanism if the model is private. No key is stored in this repository. The runtime reads `calibration.json`: the released model uses temperature **1.0**, matching the reported evaluation, and has no separately fitted temperature.

`expanded` is the evaluation reference path. `shared_prefix` and `auto` enable available prefix reuse across questions/options, including Qwen3.5's attention and recurrent states. Use `--execution-mode shared_prefix` with the inference example to select it. Hardware, batching and execution path affect numerical behavior and throughput; mode selection does not change the trained weights.

The artifact has an 8,192-token limit per expanded branch and 65,536 logical tokens per decision. The engine also defaults to 65,536 logical tokens per request. Requests are rejected rather than silently truncated. `usage` reports logical and executed tokens; `runtime.seconds` measures the model inference section. `confidence` is normalized negative entropy, not an independently calibrated correctness estimate.

## HTTP

```bash
python -m openjev.runtime.server --artifact shenjunhao/OpenJev-4B \
  --host 0.0.0.0 --port 8080 --execution-mode expanded
curl http://localhost:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/decision.json
```

The server also provides `/health` and `/v1/models`. One engine serializes model execution with a lock; request concurrency does not itself provide continuous GPU batching. To run independent workers, launch one process per GPU and distribute requests between them.

## Verified example output

Running `examples/infer.py` with the released weights on an H200 in expanded mode returned:

```json
{
  "choice": "Duplicate payment",
  "probabilities": {
    "Duplicate payment": 0.9999814629554749,
    "Unrecognized payment": 0.00001748267823131755,
    "Card delivery": 0.0000010590606507321354
  }
}
```

This is the answer excerpt for `examples/decision.json`; runtime metadata and confidence are also returned.
