---
title: "LLM \"Chat Like Me\" Project 
description: A project to create a LLM that can chat like yourself (Fine-tuning)
date: 2025-10-18 12:00:00 +0530
categories: [LLM, Chat, Project, fine-tuning]
tags: [LLM, Chat, Project, fine-tuning]
comments: false
---


### LLM "Chat Like Me" project - Part 2 - Fine-tuning

In [Part 1](https://bigghis.github.io/posts/2025-16-10-25-LLM-CHAT-LIKE-ME/), we collected and prepared a dataset from Telegram chat history. Now it's time to put that dataset to work and fine-tune an LLM that can actually chat like you.  
The idea is to fine-tune a model that can run locally on my pc with the provided dataset.  
### Training vs. Fine-tuning: An Important Distinction

Let's be clear about what we're doing here. **We're not training an LLM from scratch.** That would require millions of dollars in computational resources, massive datasets, and months of training time on specialized hardware. That's what companies like Meta, OpenAI, and Google do.

Instead, we're **fine-tuning a pre-trained model**. This means taking an existing model that already understands language, grammar, and chat patterns, and then teaching it to adopt a specific conversational style.

### Choosing a Base Model: Llama 3.1

For this project, I chose **Meta's Llama 3.1 8B** as the base model. The Llama family of models is a good choice because is open-source.  
The **8 billion parameters** version strikes a good balance between capability and computational requirements. It's small enough to fine-tune on consumer-grade GPUs but large enough to produce high-quality, contextually appropriate responses.

### QLoRA: Memory-Efficient Fine-tuning

The fine-tuning process uses a technique called **QLoRA (Quantized Low-Rank Adaptation)**. Without diving too deep into the technical details, QLoRA is a clever approach that makes fine-tuning large models practical without needing supercomputer-level hardware.

**How QLoRA works (simplified):**

Traditional fine-tuning updates all the weights in a neural network, which requires enormous memory. QLoRA instead:
1. **Quantizes** the base model to 4-bit precision, reducing memory requirements by ~75%
2. Adds small **low-rank adapter layers** that learn your specific patterns
3. Only trains these adapter layers, not the entire model

This means you can fine-tune an 8B parameter model on a single consumer GPU (like an RTX 4090 or even a rented cloud GPU) instead of needing a cluster of high-end datacenter GPUs.


### Hardware Setup for Fine-tuning

When it comes to fine-tuning LLMs, you have two main options: **cloud-based services** or **local hardware**. Cloud services like **Runpod**, **Google Colab**, or **Lambda Labs** offer the flexibility to rent powerful GPUs by the hour, making them accessible even if you don't own high-end hardware at reasonable cost.

For this project, I chose to use **my local machine** instead of cloud services. I enjoy having full control over the hardware and the freedom to experiment with different software settings and configurations. Here's my setup:

**Hardware Specifications:**
- **CPU**: 13th Gen Intel Core i7-13700K
- **RAM**: 128GB DDR4
- **GPU**: 2x NVIDIA GeForce RTX 3090 (24GB VRAM each = 48GB total)
- **OS**: Linux 6.16.8 (Endeavour OS / Arch-based)


The dual RTX 3090 setup is well-suited for this 8B model. Each card has 24GB of VRAM, which is sufficient to handle an 8B parameter model with 4-bit quantization.  
However, if you want to fine-tune **larger models** (like Llama 3.1 70B or 405B), local consumer hardware quickly becomes insufficient. For those cases, you would need to rely on cloud services like **Runpod**, **Lambda Labs**, or other providers that offer high-end datacenter GPUs (A100 80GB, H100) with significantly more VRAM. The 48GB total VRAM from two RTX 3090s simply isn't enough for larger models, even with aggressive quantization.

### Choosing a Training Framework: Axolotl

There are several frameworks available for fine-tuning LLMs. Two of the most popular are **Unsloth** and **Axolotl**. While Unsloth is known for its speed optimizations and ease of use, I chose **Axolotl** for this project for a specific reason: **multi-GPU support**.

Unsloth's support for multiple GPUs is still in an early stage and not as robust as I needed. Since I had access to a multi-GPU setup and wanted to take full advantage of it to speed up training, Axolotl was the better choice.

**Trade-offs:**
- **Axolotl** may offer fewer high-level customizations compared to Unsloth's streamlined API
- However, Axolotl provides more flexibility in configuration and better multi-GPU training support through DeepSpeed integration
- For our purposes (fine-tuning on conversational data), Axolotl's capabilities are more than sufficient

Axolotl is a well-maintained, production-ready framework with excellent documentation and strong community support, making it an ideal choice for this project.

### Setting Up the Training Configuration

Axolotl uses YAML configuration files to define all training parameters. Here's the configuration I used for this project:

```yaml
base_model: NousResearch/Meta-Llama-3.1-8B

load_in_4bit: true
strict: false

chat_template: llama3
datasets:
  - path: /path/to/training_data.json
    type: chat_template
    message_field_role: role
    message_field_content: content

dataset_prepared_path: last_run_prepared
val_set_size: 0.005
output_dir: ./outputs/lora-out

sequence_len: 4096
sample_packing: true
eval_sample_packing: false
pad_to_sequence_len: true

adapter: qlora
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_modules_to_save:
  - embed_tokens
  - lm_head
peft_use_dora: true

gradient_accumulation_steps: 2
micro_batch_size: 1
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
bf16: true
tf32: true

gradient_checkpointing: true
logging_steps: 1
flash_attention: true

warmup_ratio: 0.1
evals_per_epoch: 1
saves_per_epoch: 1
weight_decay: 0.0
```

**Key parameters explained:**

- `load_in_4bit: true`: Enables 4-bit quantization for memory efficiency (QLoRA)
- `lora_r: 64`: Rank of the LoRA adapters (higher = more capacity to learn, but more memory)
- `lora_alpha: 32`: Scaling factor for LoRA updates
- `num_epochs: 4`: Number of times to iterate through the entire dataset
- `learning_rate: 0.0002`: Controls how quickly the model adapts (too high = unstable, too low = slow learning)
- `train_on_inputs: false`: Only compute loss on assistant responses, not on user messages (more efficient training)
- `sample_packing: true`: Combines multiple conversations into single training samples for efficiency
- `flash_attention: true`: Uses optimized attention mechanism for faster training
- `peft_use_dora: true`: Uses DoRA (Weight-Decomposed Low-Rank Adaptation), an improvement over standard LoRA

### Training Process

Once the configuration is set, starting the training is straightforward with Axolotl:

```bash
accelerate launch -m axolotl.cli.train nousresearch-llama-3.1.yaml
```

For multi-GPU setups with DeepSpeed:
```bash
accelerate launch --config_file deepspeed_config.yaml -m axolotl.cli.train nousresearch-llama-3.1.yaml
```

During training, you'll see metrics like:
- **Training loss**: Should decrease over time (indicates learning)
- **Evaluation loss**: Measures performance on held-out data (helps detect overfitting)
- **Learning rate**: Follows the cosine schedule, starting high and gradually decreasing
- **Tokens per second**: Training speed metric

The training process typically takes several hours depending on your hardware and dataset size. With the Telegram dataset containing thousands of conversations, you can expect training to take anywhere from 3-8 hours on a modern GPU.

### Monitoring Training with Weights & Biases

The configuration includes integration with **Weights & Biases (wandb)**, a powerful tool for experiment tracking:

```yaml
wandb_project: chat-like-bigghis
```

This allows you to:
- Monitor training progress in real-time through a web dashboard
- Compare different training runs
- Track GPU utilization and memory usage
- Visualize loss curves and learning rate schedules
- Save training artifacts for reproducibility

### What's Next?

Once training completes, you'll have a fine-tuned model that has learned your conversational patterns. But how do you know if it actually works? How do you evaluate if the model truly "chats like you"?

In the next part of this series, we'll explore:
- How to load and use the fine-tuned model
- Evaluation techniques for conversational models
- Comparing outputs with the base model
- Real-world examples of the model in action
- Potential improvements and iterations

The journey from raw chat data to a personalized AI is almost complete!

