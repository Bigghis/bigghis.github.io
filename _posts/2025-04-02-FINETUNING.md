---
title: "Fine Tuning"
description: A fine tuning project for smart forms
date: 2025-04-02 12:00:00 +0530
categories: [Fine Tuning, Smart form ]
tags: [a.i.]
comments: false
---

## Fine Tuning

Form filling is a common task that can be automated using Large Language Models (LLMs). This project focuses on developing a system that can automatically extract relevant information from unstructured text and populate form fields intelligently.

### Project Overview


The system may works in two ways:
1. **Text Copy-Paste**: Users can copy text from any source (emails, documents, etc.) and paste it into the form.

2. **Speech-to-Text**: Users can speak the information, which gets converted to text and then pasted into the form.

### How It Works 

1. **Input Processing**: Copied text directly or convert speech to text using Speech Recognition.  

2. **LLM Processing**: The text is sent to an LLM which identifies and extracts relevant information like names, dates, addresses, and other structured data.  

3. **Form Population**: The extracted information is mapped to corresponding form fields, automatically filling the form. Users can review and edit the results.

### Demo Application

I have created a demo Angular application to test the copy-paste functionality sending the text to an LLM.


![Text Copy-Paste Demo](/assets/images/smartform.gif)
_A demonstration of the text copy-paste functionality_


The demo application can use any various LLM model to extract the information from the text, because have created backends for **OpenAI**, **Mistral** and **LLaMA** APIs in the Angular form-filler backends library.

Testing with canonical LLM models has shown very good results. While all models perform well, `gpt-3.5-turbo` from OpenAI tends to provide slightly more accurate field extraction compared to `llama3.1-8b` and `mistral-large-latest`, though both produce satisfactory results for form-filling purposes. So one can experiment with different models to find the best suitable model for the task.

You can find the demo application in GitHub project  [Smart form](https://github.com/Bigghis/smart-form).

#### System prompt to instruct LLM

For testing purposes, I used the same system prompt across all LLM backends (OpenAI, Mistral, and LLaMA) to ensure a fair comparison of their capabilities. This standardized approach helps maintain consistency in how each model processes and extracts information from the input text.






