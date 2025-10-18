---
title: "\"LLM Chat Like Me\" Project"
description: A project to create a LLM that can chat like me
date: 2025-10-16 12:00:00 +0530
categories: [LLM, Chat, Project, fine-tuning]
tags: [LLM, Chat, Project, fine-tuning]
comments: false
---

### LLM "Chat Like Me" project

This project is born out of curiosity and a desire to learn how to fine-tune large language models hands-on. It's also a fun experiment to see if I can create an AI that replicates my conversational style :) when I'm chatting with friends, colleagues, etc.

The goal is to fine-tune **Llama 3.1 8B** using my personal and group chat history from Telegram. By leveraging my actual conversations as training data, I'll capture the nuances, vocabulary, and patterns that make up my unique communication style. Through this process, I'll explore the entire pipeline of model training, from data collection to evaluation, while creating an AI finetuned model that can "chat like me."

### Why Telegram?

I chose to use **Telegram** as my data source for several reasons. First and foremost, I use it frequently for both personal conversations and group chats, which means I have a substantial amount of chat history to work with. 

I find Telegram superior to alternatives like WhatsApp, particularly for this kind of project. Telegram offers a built-in export feature that makes it easy to download your entire chat history in structured formats like JSON, which is perfect for data processing too. The platform is also more developer-friendly, with better tools and APIs for data handling, bots development and so on.  





### Data Collection

The data collection process will involve extracting chat history from both personal and group conversations on Telegram.  
**Telegram Desktop** provides a built-in feature to export all your chat history at once, which we'll use to collect our training data.

#### How to Export All Chats from Telegram

Following steps [from this article](https://www.androidpolice.com/telegram-export-chats-groups-channels-images/){:target="_blank" rel="noopener"} you can export all your chat history (personal and group chats) in JSON format.


Open telegram desktop and go to **Settings** -> **Advanced** -> **"Export Telegram Data"**   
In the export settings window a reasonable decision can be to check "Personal chats" and "Private groups" and uncheck "Photos/Videos" and "Voice messages" to not include possibly misleading data or files like photos, videos, voice messages, etc.  

![Telegram Export Settings](/assets/images/exports-telegram.png)
> **Remember:** The goal is to train a model that can chat like yourself with friends, colleagues, etc. so you need to include that chat history only!
{: .prompt-tip }

#### Understanding the Exported JSON Structure

Once the export is complete, you'll have a large JSON file (potentially hundreds of megabytes) containing all your chat history. The structure looks like this:

```json
{
 "about": "Here is the data you requested...",
 "chats": {
  "about": "This page lists all chats from this export.",
  "list": [
   {
    "name": "Weekend Plans Group",
    "type": "private_group",
    "id": 123456789,
    "messages": [
     {
      "id": 1001,
      "type": "service",
      "date": "2023-01-15T14:30:00",
      "date_unixtime": "1673792400",
      "actor": "User123",
      "actor_id": "user987654321",
      "action": "create_group",
      "title": "Weekend Plans Group",
      "members": [
       "User123",
       "Alice",
       "Bob"
      ],
      "text": "",
      "text_entities": []
     },
     {
      "id": 1002,
      "type": "message",
      "date": "2023-01-15T14:31:00",
      "date_unixtime": "1673792460",
      "from": "User123",
      "from_id": "user987654321",
      "text": "Hey everyone!",
      "text_entities": [
       {
        "type": "plain",
        "text": "Hey everyone!"
       }
      ]
     },
     {
      "id": 1003,
      "type": "message",
      "date": "2023-01-15T14:32:00",
      "date_unixtime": "1673792520",
      "from": "Alice",
      "from_id": "user111222333",
      "text": "Hi! What's up?",
      "text_entities": [
       {
        "type": "plain",
        "text": "Hi! What's up?"
       }
      ]
     }
    ]
   },
   {
    "name": "John Doe",
    "type": "personal_chat",
    "id": 987654321,
    "messages": [
     {
      "id": 2001,
      "type": "message",
      "date": "2023-02-20T10:15:00",
      "date_unixtime": "1676887500",
      "from": "User123",
      "from_id": "user987654321",
      "text": "Did you finish the project?",
      "text_entities": [
       {
        "type": "plain",
        "text": "Did you finish the project?"
       }
      ]
     },
     {
      "id": 2002,
      "type": "message",
      "date": "2023-02-20T10:16:00",
      "date_unixtime": "1676887560",
      "from": "John Doe",
      "from_id": "user444555666",
      "text": "Almost done, just some final touches",
      "text_entities": [
       {
        "type": "plain",
        "text": "Almost done, just some final touches"
       }
      ]
     }
    ]
   }
  ]
 }
}
```




**Key elements in the JSON structure:**
- **`chats.list`**: An array containing all your conversations (both personal and group chats)
- **`name`**: The name of the chat or contact
- **`type`**: Either `"personal_chat"` or `"private_group"`
- **`messages`**: An array of all messages in that conversation
- **`from`**: The sender's name
- **`text`**: The actual message content
- **`date`** and **`date_unixtime`**: Timestamps for each message

This structured format makes it relatively straightforward to parse and extract the conversation data we need for training.

  
### Data Preparation and Chat Templates

Once we have the raw JSON export, we need to transform it into a training-ready dataset. The real work begins here because there are several challenges to overcome and we need to be careful with the data quality.

**Cleaning**: Filter out group chats (unless you want multiple personalities), infrequent contacts, automated messages, and non-text content.

**Structuring**: When you chat with someone, there's typically a rapid back-and-forth exchange about a specific topic. Then hours might pass before you chat again about something completely different. To maintain logical coherence in the training data, we need to separate these distinct conversation segments. A good approach is to group messages that occur within 5 minutes or more time apart of each other as part of the same conversational topic, while treating gaps longer than an hour as natural breaks between different discussions. This will help the model learns to respond within context rather than mixing unrelated topics.

**Formatting**: The dataset will be formatted using **OpenAI's chat template**, which is a widely-adopted standard format for conversational AI. Each conversation is structured as an array of messages with clear role assignments: "user" for the other person's messages and "assistant" for your messages.

Example of a formatted conversation:
```json
{
  "messages": [
    {"role": "user", "content": "Did you finish the project?"},
    {"role": "assistant", "content": "Almost done, just some final touches"},
    {"role": "user", "content": "Great! When can I see it?"},
    {"role": "assistant", "content": "I'll send it over by tonight"}
  ]
}
```

This format is clean, human-readable, and compatible with most fine-tuning frameworks including those for Llama models.

Data quality is crucial—it directly impacts how well the model learns your conversational style.


