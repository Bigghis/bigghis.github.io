---
title: "LLM \"Chat Like Me\" Project 
description: A project to create a LLM that can chat like yourself (Fine-tuning)
date: 2025-10-18 12:00:00 +0530
categories: [LLM, Chat, Project, fine-tuning]
tags: [LLM, Chat, Project, fine-tuning]
comments: false
---


### LLM "Chat Like Me" project - Part 2 - Fine-tuning

In [Part 1](https://bigghis.github.io/posts/2025-16-10-25-LLM-CHAT-LIKE-ME/), we collected and prepared a dataset from Telegram chat history.  
Now the idea is to fine-tune a model that can run locally on my pc with the provided dataset.  
### Training vs. Fine-tuning: An Important Distinction

Let's be clear about what we're doing here. **We're not training an LLM from scratch.** That would require millions of dollars in computational resources, massive datasets, and months of training time on specialized hardware. That's what companies like Meta, OpenAI, and Google do.

Instead, we're **fine-tuning a pre-trained model**. This means taking an existing model that already understands language, grammar, and chat patterns, and then teaching it to adopt a specific conversational style.

### Choosing a Base Model: Llama 3.1

For this project, I chose **Meta's Llama 3.1 8B** as the base model. The Llama family of models is open-source and offers decent performance for conversational tasks, making it a practical choice for these experiments.  
The **8 billion parameters** version strikes a good balance between capability and computational requirements. It's small enough to fine-tune on consumer-grade GPUs but large enough to produce high-quality, contextually appropriate responses.


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

There are many frameworks available for fine-tuning LLMs. Two of the most popular are **[Unsloth](https://unsloth.ai/){:target="_blank" rel="noopener"}** and **[Axolotl](https://axolotl.ai/){:target="_blank" rel="noopener"}**. 

**Unsloth** is known for its speed optimizations and offers more granular control over the training process with extensive customization options. However, I chose **Axolotl** for this project because it better suited my needs.  
The primary factor was Axolotl's multi-GPU support through DeepSpeed integration, which has been robust for a long time. In contrast, Unsloth's multi-GPU capabilities are still in an early development stage.  
Additionally, I appreciated Axolotl's straightforward configuration approach, which is less complex compared to Unsloth's more granular settings. After setting up some additional software dependencies, Axolotl worked reliably on my dual-GPU setup without any issues. While Unsloth might offer more fine-grained control for advanced users, Axolotl's combination of simplicity, stability, and proven multi-GPU support made it the practical choice for this experiment.

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

wandb_project: chat-like-me
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

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
deepspeed: /path/to/deepspeed_configs/zero2.json
special_tokens:
  pad_token: "<|finetune_right_pad_id|>"
  eos_token: "<|eot_id|>"
```

The configuration file is organized into several logical sections, each controlling different aspects of the training process.

#### Base Model and Quantization

```yaml
base_model: NousResearch/Meta-Llama-3.1-8B
load_in_4bit: true
strict: false
```

- `base_model`: Specifies which pre-trained model to start from (Llama 3.1 8B in this case)
- `load_in_4bit`: Enables 4-bit quantization, the core of QLoRA. This reduces memory usage by ~75% compared to full precision
- `strict`: Set to false to allow more flexibility in model loading

#### Dataset and Chat Template Configuration

```yaml
chat_template: llama3
datasets:
  - path: /path/to/training_data.json
    type: chat_template
    message_field_role: role
    message_field_content: content
dataset_prepared_path: last_run_prepared
val_set_size: 0.005
```

- `chat_template`: Uses Llama 3's specific chat format to structure conversations
- `datasets.path`: Points to your prepared JSONL file with the Telegram conversations
- `type: chat_template`: Tells Axolotl to expect data in OpenAI `chat-template` format (system/user/assistant roles)
- `message_field_role` and `message_field_content`: Specifies which JSON fields contain the role and message content
- `dataset_prepared_path`: Cache directory for preprocessed data to speed up subsequent runs
- `val_set_size: 0.005`: Reserves 0.5% of data for validation (99.5% for training). This small validation set helps monitor overfitting

#### Sequence Length and Packing

```yaml
sequence_len: 4096
sample_packing: true
eval_sample_packing: false
pad_to_sequence_len: true
```

- `sequence_len: 4096`: Maximum context length in tokens. Llama 3.1 supports up to 128K, but 4096 is sufficient for most chat conversations and more memory-efficient
- `sample_packing: true`: Combines multiple short conversations into single training batches to maximize GPU utilization. Instead of wasting tokens on padding, multiple conversations fill the 4096 token window
- `eval_sample_packing: false`: Disables packing during evaluation for cleaner metrics
- `pad_to_sequence_len: true`: Ensures consistent sequence lengths for efficient batch processing

#### LoRA/QLoRA Configuration

```yaml
adapter: qlora
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_modules_to_save:
  - embed_tokens
  - lm_head
peft_use_dora: true
```

This section defines how the model will be adapted:

- `adapter: qlora`: Uses [QLoRA (Quantized LoRA)](https://arxiv.org/abs/2305.14314){:target="_blank" rel="noopener"} technique
- `lora_r: 64`: The rank of the LoRA adapter matrices. Higher values (like 64) give the model more capacity to learn patterns but require more memory. Typical values range from 8 to 128
- `lora_alpha: 32`: Scaling factor that controls how much the adapter influences the base model. The ratio `lora_alpha/lora_r` determines the learning strength
- `lora_dropout: 0.05`: Applies 5% dropout<sup>(1)</sup> to prevent overfitting<sup>(2)</sup> in the adapter layers
- `lora_target_linear: true`: Applies LoRA to all linear layers in the transformer blocks (attention and feed-forward)
- `lora_modules_to_save`: In addition to LoRA adapters, these modules are fully fine-tuned. The embedding (`embed_tokens`) and output (`lm_head`) layers often benefit from full fine-tuning, especially when working with specific vocabulary or domains
- `peft_use_dora: true`: Enables DoRA (Weight-Decomposed Low-Rank Adaptation), a recent improvement over standard LoRA that separates magnitude and direction updates for better performance

<sup>(1)</sup> *[dropout](https://bigghis.github.io/AI-appunti/guide/regularizations/dropout.html?highlight=dropout#dropout){:target="_blank" rel="noopener"} is a regularization technique that randomly "turns off" a subset of neurons during training, helping to prevent overfitting by forcing the network to learn more robust features.*

<sup>(2)</sup> *[Overfitting](https://bigghis.github.io/AI-appunti/guide/generics.html?highlight=overfitting#generalizzazioni-del-comportamento-delle-reti-neurali){:target="_blank" rel="noopener"} occurs when a model adapts too closely to the details and noise in the training data, compromising its ability to generalize. The model learns not only the underlying patterns but also random fluctuations and anomalies specific to the training set, resulting in excellent performance on training data but poor performance on new, unseen data.*

#### Training Hyperparameters

```yaml
gradient_accumulation_steps: 2
micro_batch_size: 1
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002
```

- `gradient_accumulation_steps: 2`: Accumulates gradients<sup>(3)</sup> over 2 forward passes before updating weights. This simulates a larger batch size (effective batch size = micro_batch_size × gradient_accumulation_steps × num_gpus = 1 × 2 × 2 = 4) without requiring more memory
- `micro_batch_size: 1`: Processes one conversation at a time per GPU. With sample packing enabled, this could contain multiple conversations packed into 4096 tokens
- `num_epochs: 4`: Iterates through the entire dataset 4 times. More epochs can lead to better adaptation but risk overfitting
- `optimizer: adamw_bnb_8bit`: Uses 8-bit AdamW optimizer (from bitsandbytes library) for memory efficiency
- `lr_scheduler: cosine`: Learning rate follows a cosine curve, starting at the specified rate and gradually decreasing to near zero
- `learning_rate: 0.0002`: Starting learning rate. This is relatively standard for LoRA fine-tuning

<sup>(3)</sup> *[Gradient accumulation](https://colab.research.google.com/drive/102AQrQf0YJqWTGH0aKDnmZXftSHTQNS_?usp=sharing){:target="_blank" rel="noopener"} is a technique to avoid running out of VRAM during training. Normally, the model weights are updated after every batch using the calculated gradients. With gradient accumulation, instead of updating immediately, the gradients are accumulated (summed) over multiple batches. The model weights are only updated after a specified number of iterations. This allows training with larger effective batch sizes without requiring additional memory.*

#### Training Efficiency Settings

```yaml
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

- `train_on_inputs: false`: Only computes loss on assistant responses, not on system prompts or user messages. This focuses learning on your conversational style
- `bf16: true`: Uses bfloat16 precision for training, which balances memory efficiency with numerical stability
- `tf32: true`: Enables TensorFloat-32 on compatible GPUs (like RTX 3090) for faster matrix operations
- `gradient_checkpointing: true`: Trades computation for memory by recomputing activations during backward pass instead of storing them all. Essential for training larger models
- `logging_steps: 1`: Logs training metrics every step (useful for monitoring but can slow things down slightly)
- `flash_attention: true`: Uses Flash Attention 2, a highly optimized attention implementation that significantly speeds up training
- `warmup_ratio: 0.1`: Gradually increases learning rate from 0 to the target over the first 10% of training steps to prevent instability
- `evals_per_epoch: 1`: Evaluates on validation set once per epoch
- `saves_per_epoch: 1`: Saves a checkpoint once per epoch
- `weight_decay: 0.0`: No L2 regularization. LoRA's low rank already provides implicit regularization

#### DeepSpeed Configuration

```yaml
deepspeed: /path/to/deepspeed_configs/zero2.json
```

DeepSpeed is a distributed training library that enables efficient multi-GPU training. 
The **ZeRO (Zero Redundancy Optimizer)** technique comes in three stages:

- **ZeRO-1**: Partitions optimizer states across GPUs. Provides modest memory savings
- **ZeRO-2**: Partitions both optimizer states and gradients. This is what we're using—it provides a good balance between memory savings and communication overhead, making it ideal for 2-GPU setups
- **ZeRO-3**: Partitions model parameters, optimizer states, and gradients. Maximum memory savings but requires more inter-GPU communication. Best for training very large models across many GPUs (8+)

In my setup I used an[ADAM] optimizer(https://bigghis.github.io/AI-appunti/guide/optimizations/adamoptimizations.html?highlight=adam#adam--adaptive-moment-estimation){:target="_blank" rel="noopener"}. It state maintains two momentum states for each model parameter—a moving average of gradients (first moment) and a moving average of squared gradients (second moment). This optimizer state typically requires about 8 bytes per model weight, which adds substantial memory overhead even with 8-bit quantization.


ZeRO-1 only partitions ADAM optimizer states across GPUs, but in my case this wasn't enough to avoid memory saturation. Switching to ZeRO-2 resolved the issue because it partitions both ADAM optimizer states and gradients across the two RTX 3090s.


#### Special Tokens

```yaml
special_tokens:
  pad_token: "<|finetune_right_pad_id|>"
  eos_token: "<|eot_id|>"
```

- `pad_token`: Token used for padding sequences to the same length
- `eos_token`: End-of-turn token specific to Llama 3.1's chat format

#### Weights & Biases Integration

```yaml
wandb_project: chat-like-me
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:
```

These settings configure [Weights & Biases](https://wandb.ai/){:target="_blank" rel="noopener"}, a tool for tracking and logging the training process, including metrics like loss, learning rate, and GPU utilization.

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

The training process typically takes several hours depending on your hardware and dataset size. 


