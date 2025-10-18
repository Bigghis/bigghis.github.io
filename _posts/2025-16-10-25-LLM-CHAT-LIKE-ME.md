---
title: "\"LLM Chat Like Me\" Project"
description: A project to create a LLM that can chat like me
date: 2025-10-16 12:00:00 +0530
categories: [LLM, Chat, Project, fine-tuning]
tags: [LLM, Chat, Project, fine-tuning]
comments: false
---

### LLM Chat Like Me project

This project is born out of curiosity and a desire to learn how to fine-tune large language models hands-on. It's also a fun experiment to see if I can create an AI that replicates my conversational style :) when I'm chatting with friends, colleagues, etc.

The goal is to fine-tune **Llama 3.1** using my personal and group chat history from Telegram. By leveraging my actual conversations as training data, I'll capture the nuances, vocabulary, and patterns that make up my unique communication style. Through this process, I'll learn the entire pipeline of model training—from data collection to evaluation—while creating an AI that can "chat like me."

### Why Telegram?

I chose to use **Telegram** as my data source for several reasons. First and foremost, I use it frequently for both personal conversations and group chats, which means I have a substantial amount of chat history to work with. 

I find Telegram superior to alternatives like WhatsApp, particularly for this kind of project. Telegram offers a built-in export feature that makes it easy to download your entire chat history in structured formats like JSON, which is perfect for data processing too. The platform is also more developer-friendly, with better tools and APIs for data handling, bots development and so on.  





### Data Collection

The data collection process will involve extracting chat history from both personal and group conversations on Telegram.  
**Telegram Desktop** provides a built-in feature to export all your chat history at once, which we'll use to collect our training data.

#### How to Export All Chats from Telegram

Following steps [from this article](https://www.androidpolice.com/telegram-export-chats-groups-channels-images/) you can export all your chat history (personal and group chats) in JSON format.


Open telegram desktop and go to **Settings** -> **Advanced** -> **"Export Telegram Data"**   
In the export settings window a reasonable decision can be to check "Personal chats" and "Private groups" and uncheck "Photos/Videos" and "Voice messages" to not include possibly misleading data or files like photos, videos, voice messages, etc.  

![Telegram Export Settings](/assets/images/exports-telegram.png)
> **Remember:** you want to train a model that can chat like you with friends, colleagues, etc. so you need to include that chat history only!
{: .prompt-tip }


### Data Preparation and Chat Templates

The raw chat data will need to be processed and converted into training-ready format:
- Parsing the exported Telegram data structure
- Creating proper chat templates compatible with Llama 3.1's expected format
- Handling multi-turn conversations and context windows
- Filtering and cleaning the dataset (removing sensitive information, handling special characters, etc.)
- Structuring conversations in the appropriate prompt-response format

### Technical Details

More details about the implementation, including code examples and technical decisions, will be documented as the project progresses.
