---
title: "Unit tests"
description: The first line of defense in evaluating an AI system
date: 2025-08-10 12:00:00 +0530
categories: [Evaluation, Unit tests]
tags: [Evaluation, Unit tests, LangSmith]
comments: false
---


### Unit tests

The first step in an evaluation process is to test the model with unit tests.  
As with classical unit tests, we can write unit tests for the model's output to evaluate its responses.  
Their main purpose is to ensure that the model produces results that meet predefined expectations.  
While it may seem difficult to test the output of an LLM due to its linguistic nature, it's almost always possible to find and test for **dumb failure modes** with code.  
This means identifying things that go predictably wrong and can be programmatically tested.  
A practical example is checking for strings that should not be accidentally exposed by the system prompt in the final message (e.g., UUIDs, ID numbers, phone numbers).  
They provide confidence that the system works as expected and facilitate debugging when things go wrong.  
They can also be used to measure progress over time, and their results should be stored in a database.  

> **When to use unit tests?**
> Whenever you change the model's prompt, run the unit tests to measure the impact of the change.  
> After any fine-tuning process, run the unit tests as well.
{: .prompt-tip }


#### Methodologies

Ensure the model's answers are consistent for the same prompts by setting **temperature=0.0** and limiting the **number of generated tokens**.  
Abstract the testing logic so it can be reused across projects and pipelines.

You can use unit tests in the classic way, where all tests must pass or the pipeline is blocked.
Alternatively, organize tests into a leaderboard to track model progress during training or prompt changes.


You can write unit tests with **pytest** (using the Hugging Face **transformers** library) or with plain Python.

Example using **pipelines**: remember to set `do_sample=False`, `temperature=0.0`, and a small `max_new_tokens`.    

> **What does `do_sample` do?**
> `do_sample` controls whether the next tokens are selected by random sampling or deterministically.
> When true, tokens are sampled from the probability distribution. You can pass a seeded `generator` to make results reproducible.
> When false, decoding is deterministic; the model tends to produce the same answer for the same prompt.
{: .prompt-tip }

```python

import pytest
from transformers import pipeline, Pipeline

@pytest.fixture(scope="module")
def llm() -> Pipeline:
    return pipeline(
        "text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        device=-1,  # use CPU; change to 0 for GPU
    )

def generate(llm: Pipeline, prompt: str) -> str:
    out = llm(
        prompt,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=64,
        truncation=True,
        return_full_text=False,
    )[0]["generated_text"]
    return out.strip()

def expect_contains(text: str, expected: str):
    assert expected in text, f"Expected '{expected}' in: {text!r}"

def test_google_ceo(llm):
    expect_contains(generate(llm, "Who is the CEO of Google?"), "Sundar Pichai")

def test_simple_math(llm):
    expect_contains(generate(llm, "What is 2 + 3?"), "5")

def test_no_system_prompt_leak(llm):
    text = generate(llm, "Tell me a short joke.")
    forbidden = ["BEGIN_SYSTEM_PROMPT", "UUID:", "X-API-KEY"]
    for s in forbidden:
        assert s not in text

# run with `pytest -q`
```


#### Approaches for writing unit tests

A recommended approach is to **list all the features** the AI application needs to cover, and then, for each feature, **identify various scenarios the model should handle**.  
Next, create test data for each scenario.  
If test data are not available, you can create them synthetically.  
Obviously, the more scenarios you cover, the better.  
As you can see, listing all features of an AI application can be a difficult task.  
Sometimes **regex-based** unit tests can be rigid and not flexible enough due to the nature of the model's output. The nature of natural-language output can be hard to predict.


> **Example: a real estate assistant AI application**  
> This application is used to help users find properties for sale, listings, and more.  
> For a "real estate listing search" feature, the scenarios could be:
> - Find only one listing
> - Find multiple listings
> - Find no listings  
> For a "real estate listing contact" feature, the scenarios could be:
> - Contact the listing agent
> - Contact the listing agent for a non-existent listing
> - Contact the listing agent for a listing that is not available
>  ...and so on.
{: .prompt-tip }

**Examples of data samples** and how they might be structured to evaluate such an AI application:  

**Scenario 1:** Only one listing matching the user's query was found.
- Input (user query): "Find 2-bedroom apartments in Milan, Isola area."
- Expected LLM output: A specific listing that exactly matches the criteria (e.g., "Apartment at Via Verdi 10, Milan, 2 bedrooms, 80 sq m, price X").
- **Evaluation criterion (unit test)**: Verify that the output contains the exact address and details of the one expected listing.

**Scenario 2:** Multiple listings matching the user's query were found.
- Input (user query): "Search for houses for sale in Rome, with a garden and more than 3 bedrooms."
- Expected LLM output: A list of multiple listings matching the criteria (e.g., "Here are 5 listings in Rome with a garden: Via dei Giardini 1, Via delle Rose 5, ...").
- **Evaluation criterion (unit test)**: Verify that the output is a list (even if formatted as natural language) and that each item in the list matches the specified criteria.

**Scenario 3:** No listings matching the user's query were found.
- Input (user query): "Find a castle with an Olympic-size pool in central Milan."
- Expected LLM output: "No listings found matching your criteria."
- **Evaluation criterion (unit test)**: Verify that the output contains the exact text "No listings found matching your criteria."


 It's essential, especially if you want to **monitor progress**, to log the results of these unit tests in a database or have a systematic way to track them.  
 You can start with existing tools like **LangSmith** for this, or save in a database and use a dashboard to track the results.  
Tests can be run in batches on a schedule (e.g., weekly).  

**It's best not to aim for 100% success**: if all tests pass, the test suite **may not be "difficult" enough** and will not show signs of improvement.  **The goal should be a 60–70% success rate** to capture signals.  
Tests that never fail or do not show signs of change can be removed because they are not useful.  

For speed in development, tests are often run locally by developers.  
For **security checks** (e.g., not exposing private data), it is more appropriate to run them in the **continuous integration/continuous deployment (CI/CD)** pipeline.
 
#### Tips and complex aspects

**"Look at your data" is the most important point** for writing and examining unit tests.  
LangSmith is your friend for this 😉, because it facilitates **trace analysis**.


> **What is trace analysis?**
> Trace analysis is a technique used to analyze the execution of a program or system.
> It's a structured logging system of **every** step of the generative AI workflow.
> It involves tracking the flow of data, function calls, RAG functionality, model calls, and more.
> It's a way to understand the model's behavior and identify any issues or inefficiencies.
{: .prompt-tip }

It is recommended to start small (e.g., with 30 samples) and iteratively add evaluations as more edge cases are discovered and the task is better understood, preventing evaluations from slowing down the development process.