---
title: "Fine Tuning"
description: A fine tuning project for smart forms
date: 2025-04-02 12:00:00 +0530
categories: [Fine Tuning, Smart form ]
tags: [a.i.]
comments: false
---

### Project Overview

Form filling is a common task that can be automated using Large Language Models (LLMs). This project focuses on developing a system that can automatically extract relevant information from unstructured text and populate form fields intelligently.


The system may works in two ways:
1. **Text Copy-Paste**: Users can copy text from any source (emails, documents, etc.) and paste it into the form.

2. **Speech-to-Text**: Users can speak the information, which gets converted to text and then pasted into the form.

### How It Works 

1. **Input Processing**: Copied text directly or convert speech to text using Speech Recognition.  

2. **LLM Processing**: The text is sent to an LLM which identifies and extracts relevant information like names, dates, addresses, and other structured data.  

3. **Form Population**: The extracted information is mapped to corresponding form fields, automatically filling the form. Users can review and edit the results.

### Demo Application

I have created a demo Angular application to test the copy-paste functionality sending the text to an LLM, 


![Text Copy-Paste Demo](/assets/images/smartform.gif)
_A demonstration of the text copy-paste functionality_


The demo application can use any various LLM model to extract the information from the text, because have created backends for **OpenAI**, **Mistral** and **LLaMA** APIs in the Angular form-filler backends library, 
inspired by the [smart-form-filler](https://github.com/thinktecture-labs/smart-form-filler) project.

Testing with canonical LLM models has shown very good results. While all models perform well, `gpt-3.5-turbo` from OpenAI tends to provide slightly more accurate field extraction compared to `llama3.1-8b` and `mistral-large-latest`, though both produce satisfactory results for form-filling purposes. So one can experiment with different models to find the best suitable model for the task.

You can find the demo application in GitHub project  [smart form](https://github.com/Bigghis/smart-form).

#### System prompt

For testing purposes, I used the same system prompt across all LLM backends (OpenAI, Mistral, and LLaMA) to ensure a fair comparison of their capabilities. This standardized approach helps maintain consistency in how each model processes and extracts information from the input text.

the prompt is the same generic prompt taken from the [smart-form-filler](https://github.com/thinktecture-labs/smart-form-filler) project.


**system prompt** pseudocode:
```python
    "role": "system",
    "content": """Each response line matches the following format:FIELD identifier^^^value
    Give a response with the following lines only, with values inferred from USER_DATA:
    FIELD firstName^^^The firstName of type string
    FIELD lastName^^^The lastName of type string
    FIELD phoneNumber^^^The phoneNumber of type string
    FIELD addressLine1^^^The addressLine1 of type string
    FIELD addressLine2^^^The addressLine2 of type string
    FIELD email^^^The email of type string
    FIELD city^^^The city of type string
    FIELD state^^^The state of type string
    FIELD zip^^^The zip of type string
    FIELD country^^^The country of type string
    FIELD birthDate^^^The birthDate of type string
    FIELD birthPlace^^^The birthPlace of type string
    FIELD birthCountry^^^The birthCountry of type string
    END_RESPONSE
    Do not explain how the values were determined. # to not divagate from the task
    Trying to deduce state where do he lives.   # this is a hint for the LLM to deduce the living state
    Trying to deduce birth country also.       # this is a hint for the LLM to deduce the birth country
    For fields without any corresponding information in USER_DATA, use value NO_DATA.

```

The fields list can be build dynamically reading the form fields, so can be adapted to any form type.

#### User prompt

the **user prompt** is the text copied from the user or the speech converted to text, and can be formatted in this simple way:

```python
"role": "user",
"content": """USER_DATA: <text>""" # Sono Mario Rossi e vivo a Milano da tre anni...

```

#### LLM response

After sending the prompts to the LLM, via API, the response will be in the following format:

```python
"role": "assistant",
"content": """FIELD firstName^^^Mario
FIELD lastName^^^Rossi
FIELD addressLine1^^^....
``` 

So it's straightforward to parse the response and populate the form fields!

### Using private/local LLM models

The system can be extended to use private/local LLM models by adding a new backend for the local model and a new form-filler backend.



> **Why use private/local LLM models?** 
> For privacy reasons, because user data is not sent to the cloud, so it's private and secure.
{: .prompt-tip }









