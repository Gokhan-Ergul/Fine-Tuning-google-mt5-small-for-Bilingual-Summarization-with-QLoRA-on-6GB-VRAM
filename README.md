# Fine-Tuning google/mt5-small for Bilingual Summarization with QLoRA on 6GB VRAM

This repository documents the process of fine-tuning the `google/mt5-small` (300M parameters) model for a bilingual (Turkish & English) summarization task. The primary challenge and focus of this project was to successfully execute this task on consumer-grade hardware with limited VRAM: an **NVIDIA GeForce RTX 3050 Laptop GPU with 6GB of VRAM**.

This was made possible by implementing **QLoRA** (Quantized Low-Rank Adaptation).

The notebook (`Summarization.ipynb`) is a detailed log of the entire journey, including:
1.  Setting up the QLoRA configuration.
2.  Combining and processing bilingual datasets.
3.  Successfully debugging a critical `OverflowError` and `NaN` loss during training.
4.  Diagnosing the root cause of a "model collapse" (`<extra_id_0>` output) in the final model.

---
## Final Model Status: A Learning Experience

The final 8-epoch trained model (available on the Hub at [gokhanErgul/mt5-small-finetuned-tr-en](https://huggingface.co/gokhanErgul/mt5-small-finetuned-tr-en)) is **functionally broken**.

While the training *ran to completion* after fixing several stability bugs, a critical error in the initial data processing led to the model learning incorrect behavior. The model produces repetitive, nonsensical text starting with `<extra_id_0>`.

**This repository is primarily a successful case study on *debugging the QLoRA training process* on limited hardware, and a diagnostic log of *how data processing can break a model*.** See the "Final Diagnosis" section for a full breakdown.

---
##  Core Technologies
* **Model:** `google/mt5-small`
* **Fine-Tuning Technique:** **QLoRA**
* **Quantization:** `bitsandbytes` (4-bit `nf4` quantization)
* **Adaptation:** `peft` (Low-Rank Adaptation - LoRA)
* **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
* **Datasets:**
    * `reciTAL/mlsum` (Turkish)
    * `abisee/cnn_dailymail` (English)
* **Metrics:** ROUGE and BERTScore

---
##  Installation

The following libraries are required to replicate this project:
```bash
pip install transformers
pip install peft
pip install bitsandbytes
pip install datasets
pip install evaluate
pip install bert-score
pip install nltk
pip install sentencepiece
pip install accelerate
```


#  Project Workflow & Debugging Journey

This project was not a simple run.  
The main value lies in the **debugging process** that was required to get the model to train for **8 epochs without crashing**.

---

##  Part 1: Load Data
- **Datasets Used:**
  - `abisee/cnn_dailymail` (English)
  - `reciTAL/mlsum` (Turkish)

---

##  Part 2: Setup QLoRA

Define a `BitsAndBytesConfig` to load the model in **4-bit (nf4 type)** for efficient fine-tuning.

---

##  Part 3: The Debugging Journey – Fixing OverflowError and NaN

The initial training attempt **failed at the very first evaluation step** with:


---

###  Problem

A combination of **three issues** was identified:

1. **Gradient Explosion:**  
   Using `bf16=True` (or `fp16=True`) with 4-bit quantization is inherently unstable.  
   During training, the gradients would “explode” to infinity (`Inf`).

2. **Model Poisoning:**  
   These `Inf` values would **poison the model’s weights**, causing all subsequent outputs (predictions) to become `NaN` (Not a Number).

3. **Metrics Crash:**  
   The `compute_metrics` function would receive these `NaN` values.  
   Then, the line:
   ```python
   tokenizer.batch_decode(predictions, ...)

##  Solution (A Three-Part Fix)

### 1. Stabilize Training
Add `max_grad_norm = 1.0` to `Seq2SeqTrainingArguments` to prevent gradient explosion.

### 2. Robust Metrics
Make the `compute_metrics` function tolerant to **NaN** or corrupted predictions.

### 3. Prevent OOM on Eval
Force **BERTScore** to run on the **CPU** to prevent it from competing with the main model for VRAM.

#  Final Diagnosis: Why the Trained Model is Broken

## Observed Symptoms
After 8 epochs, the training completed, but the model produces garbage output during inference. Outputs often include tokens like `<extra_id_0>` and repetitive, unrelated text (e.g., "Türkiye, resmî adıyla....").

---

##  Root Cause: Label Truncation
The core problem was an incorrect setting in the data preprocessing step for the **target labels**:

* `max_target_length = 30`
* `truncation = True`

This code took all target summaries longer than 30 tokens and cut off the end. This process also **cut off the `</s>` (end-of-sequence) token** from almost every single label in the dataset.

---

##  The Result
The model was trained for 8 epochs on thousands of examples, but it **never learned how to properly stop a sentence**. It was never shown a complete, finished summary example that ended with the `</s>` token.

---

##  Why `<extra_id_0>`?
When the model is asked to generate text, it panics because it doesn't know how to finish the sequence.

1.  It attempts to perform its fine-tuned task (summarization) but fails because it has no concept of "done."
2.  It **reverts to its original pre-training task** (fill-in-the-blanks / masked language modeling).
3.  This pre-training task uses `<extra_id_0>`, `<extra_id_1>`, etc., as standard placeholders.
4.  The model outputs this token and then gets stuck in a repetitive loop, as it still doesn't know when to stop.

