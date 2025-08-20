---
title: "Evaluation"
description: A summary of the different evaluation techniques
date: 2025-08-07 12:00:00 +0530
categories: [Evaluation]
tags: [Evaluation, Unit tests, LLM as a Judge, Human Evaluation]
comments: false
mermaid: true
---


### Evaluation

Evaluation is the process of assessing the performance of a model. It's a **crucial step** in the development of a model, as it allows us to understand how well the model is performing and to identify areas for improvement, starting with the model's data.
It guarantees model reliability through the iterative process of training and evaluation.  
They help you understand whether the model is good enough to be used in production and help you debug the model later when things go wrong.  
Since they are important, they **should be created early** in the development process, and expanded as the model evolves.

Evaluation is a process that can involve different techniques and workflows.  

Examples:
How to evaluate a code-generation model (SQL, Python, etc.)?  
Run the generated code in a sandbox **(execution evaluation)** and compare the output with the expected output.  

> In production, customers may reject an LLM's answers, and it's not easy to deduce the reason for their rejection. Evaluations must account for the fact that clients may ask questions differently from those conducting the evaluations and it's crucial to identify many use cases, trying to replicate the questions asked by customers
{: .prompt-tip }



Generally speaking, evaluation consists of different techniques; we can divide them into three categories:

#### Unit Tests 
[Unit tests](https://bigghis.github.io/posts/UNIT-TESTS/) are the first step in an evaluation process. 
They should be used to quickly identify **dumb failure modes**—things that go predictably wrong—and test them programmatically.
They can be used every time the prompt is changed or data is added or modified.


#### LLM as a Judge
LLM as a judge is a type of test performed by an LLM to evaluate the performance of another LLM. The judge LLM receives a prompt and a response and should be able to judge whether the response is correct.
It's a scalable way to evaluate the model's performance and is popular in the industry.  
It's essential to measure the judge LLM's correlation with human judgment.



#### Human Evaluation
It involves having the model's output directly examined by a person or a team who evaluate its quality.  
The human judge is the most reliable way to evaluate the model's performance, and sometimes it's the only way to assess the model's performance.  
It's expensive, time-consuming, and poorly scalable.
It can be affected by the human judge's bias.

#### The Evaluation workflow


```mermaid
flowchart LR
  subgraph Evaluation Cycle
    I["LLM innovations"]
    U["Unit tests"]
    L["Logging traces"]
    E["Eval & curation (automate)"]
    P["Prompt engineering"]
    F["Fine-tune w/ curated data"]
    M["Improve model"]
  end

  I --> U
  I --> L
  U --> E
  L --> E
  E --> P
  E --> F
  P --> M
  F --> M
  M --> I

  UF[/"User feedback / A-B tests"/]
  R[/"Reward model / Auto-eval"/]

  E -.-> UF
  E <-.-> R

  classDef emphasized fill:#e9f7ef,stroke:#2ecc71,stroke-width:2px,color:#0b6623;
  class I,U,L,E,P,F,M emphasized;
```

Notes:
- Unit tests and traces surface regressions and dumb failures early.
- Eval & curation drive both prompting and fine-tuning.
- Model improvements loop back to kick off the next iteration; feedback and reward models extend automation.
