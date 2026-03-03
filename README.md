# Nanbeige4-3B Blind Spot Benchmark

## Overview

This dataset contains 10 diverse prompts where the base language model\
**Nanbeige/Nanbeige4-3B-Base** produced incorrect or misleading outputs.

The goal of this dataset is to systematically identify and categorize
the model's blind spots, including:

-   Temporal reasoning errors\
-   Semantic misinterpretation\
-   False premise hallucination\
-   Generation instability\
-   Sequence misinterpretation\
-   Factual hallucination\
-   Forced choice bias\
-   Termination truncation\
-   Logical reasoning failures\
-   Nested conditional breakdown\
-   Geographic misinformation\
-   Lexical constraint violations\
-   Token repetition loops

This benchmark is designed as a qualitative failure analysis dataset
rather than a large-scale evaluation suite.

------------------------------------------------------------------------

## Model Tested

Model: https://huggingface.co/Nanbeige/Nanbeige4-3B-Base\
Type: Base (not instruction-tuned)\
Parameters: 3B\
Architecture: Causal Language Model

------------------------------------------------------------------------

## How the Model Was Loaded

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Nanbeige/Nanbeige4-3B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

prompt = "Write a sentence without using any words that appear in the Dictionary."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

output_ids = model.generate(
    input_ids,
    max_new_tokens=500,
    temperature=0.7,
    do_sample=True
)

resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
print(resp)
```

All model outputs stored in this dataset were generated using this
configuration.

------------------------------------------------------------------------

## Dataset Structure

Each entry contains the following fields:

  Field               Description
  ------------------- -----------------------------
  `input`             Prompt fed to the model
  `expected_output`   Correct or intended answer
  `model_output`      Actual model response
  `blind_spot`        Description of the failure
  `category`          High-level failure category

------------------------------------------------------------------------

## Identified Blind Spots

### 1. Temporal Reasoning

-   Leap-year birthday miscalculation
-   Off-by-one date reasoning errors

### 2. Logical Word Problems

-   Misinterpretation of ownership vs state change (egg riddle)

### 3. Factual Hallucination

-   Incorrect claims about global copper production
-   Fabricated historical events (FIFA host country)

### 4. State Tracking Failure
-   State tracking failure
-   Multi step planning error

### 5. Counting Reason Failure
-   Shared-sibling counting error

### 6. Multi-Step Arithmetic

-   Fraction miscalculation
-   Sequential reasoning collapse

### 7. Geography Error
-   Geographical misinformation
-   Hallucinated historical event.

### 8. Forced Choice Bias

-   Mild Generation Instability
-   Overjustification

### 9. Formal Logic Failures

-   Negation and syllogism breakdown (bloops/razzies/ziffs)

### 10. Generation Pathologies

-   Infinite token repetition ("separate, separate...")
-   Degeneration under impossible constraints

------------------------------------------------------------------------

## What This Reveals About the Model

The Nanbeige4-3B-Base model shows the following systemic weaknesses:

-   Limited symbolic reasoning robustness\
-   Weak global constraint tracking\
-   Susceptibility to hallucination under factual prompts\
-   Degeneration under adversarial lexical constraints\
-   Shallow multi-step reasoning capability

These behaviors are consistent with expectations for a base 3B parameter
model without instruction tuning or reasoning fine-tuning.

------------------------------------------------------------------------

## How Could These Errors Be Fixed?

### Recommended Fine-Tuning Data

  Weakness                  Recommended Dataset Type
  -----------------------   --------------------------------------
  Temporal reasoning        Date reasoning QA datasets
  Logical Word Problems     Math word problem datasets
  Factual Hallucination     High-quality fact-checked QA datasets (Natural Questions, TriviaQA)
  State Tracking Failure    Multi-step planning datasets (BABI, ReClor reasoning tasks)
  Counting Reason Failure   Multi-step sequential reasoning datasets
  Multi-Step Arithmetic     GSM8K (multi-step math problems)
  Geography Error           GeoQA datasets (e.g., GeogQ, WikiData facts)
  Forced Choice Bias        Multiple-choice reasoning datasets (RACE, OpenBookQA, ARC)
  Formal logic              Synthetic logic reasoning corpora
  Generation Pathologies    High-quality, human-edited text examples

------------------------------------------------------------------------

## How to Assemble Such a Dataset

Potential sources:

-   GSM8K (math reasoning)
-   StrategyQA (multi-hop reasoning)
-   BIG-Bench logical subsets
-   Synthetic syllogism generators
-   Date & calendar synthetic prompt generators
-   Fact-checked QA pairs from Wikipedia
-   Adversarial lexical constraint generation tasks

Synthetic data can also be programmatically generated for:

-   Leap year reasoning
-   Fraction sequencing
-   Formal logic truth evaluation
-   Constraint-following generation tasks

------------------------------------------------------------------------

## Estimated Dataset Size for Effective Fine-Tuning

  Domain                 Estimated Size Needed
  ---------------------- -----------------------
  Arithmetic reasoning   10k--50k examples
  Logical reasoning      10k--30k examples
  Temporal reasoning     5k--20k examples
  Lexical constraints    20k+ examples
  Factual QA grounding   50k+ examples

A combined dataset of approximately **100k--300k high-quality examples**
would likely be required for meaningful improvement across all domains.

------------------------------------------------------------------------

## Intended Use

This dataset is intended for:

-   Qualitative model evaluation\
-   Blind spot analysis\
-   Small-model reasoning research\
-   Fine-tuning diagnostic benchmarking\
-   Academic experimentation

It is not intended as a statistically representative benchmark.

------------------------------------------------------------------------

## Limitations

-   Only 10 examples (small diagnostic sample)
-   Not statistically representative
-   Model outputs may vary with different decoding parameters
-   Some prompts are intentionally adversarial

Future versions may expand to 100+ examples.

------------------------------------------------------------------------

## Final Notes

This dataset demonstrates that even modern 3B parameter models:

-   Struggle with structured reasoning\
-   Hallucinate under factual uncertainty\
-   Fail at global lexical constraints\
-   Collapse under adversarial generation conditions

Systematic blind spot analysis is essential for:

-   Understanding model limitations\
-   Designing fine-tuning curricula\
-   Building safer and more reliable language systems
