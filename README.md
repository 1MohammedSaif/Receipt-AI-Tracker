# AI-Powered Receipt Tracking System (Donut Transformer)

## Project Overview
As a Data Science student, I built this end-to-end pipeline to automate expense tracking. The system uses a **Vision-to-Structured-Text** approach, bypassing traditional OCR limitations by using the **Donut** (Document Understanding Transformer) architecture.

### Architecture

![Donut Transformer Diagram](/donut_architecture.jpg)

Unlike traditional OCR systems that require separate text detection and recognition modules, Donut (Document Understanding Transformer) operates as a unified **End-to-End** system:

1. **Vision Encoder (Swin Transformer):** Processes the raw input image and converts it into a tensor of high-dimensional embeddings (batch_size, seq_len, hidden_size).
2. **Text Decoder (BART):** An autoregressive decoder that takes those embeddings and generates structured text (JSON) one token at a time, conditioned on the visual features provided by the encoder.

This OCR-free approach reduces error propagation and allows the model to understand the **spatial layout** of the receipt—which is why it can identify a "Total Price" even if the text is slightly bent.

## Tech Stack
* **Deep Learning:** Hugging Face Transformers, PyTorch
* **Model:** `naver-clova-ix/donut-base-finetuned-cord-v2`
* **Data Handling:** Pandas, OpenPyXL
* **Environment:** VS Code, Virtualenv

## Key Features
* **OCR-Free Extraction:** Uses a Swin-Transformer encoder to read image pixels directly into JSON.
* **Automated Logging:** Automatically appends extracted data to an Excel tracker (`receipt_log.xlsx`).
* **Cross-Platform:** Optimized for both GPU (CUDA) and CPU environments.

## How to Run
1. Place receipt images in the `data/` folder.
2. Run `python app.py`.git add.
3. Check `receipt_log.xlsx` for the structured output.

## !!!! Model Bias & Limitations
As part of my analysis, I identified a specific **domain bias** in the underlying model:
* **Training Source:** The model was fine-tuned on the [CORD dataset](https://github.com/clovaai/cord) (Consolidated Receipt Dataset), which primarily consists of restaurant and retail receipts.
* **Impact:** When processing specialized documents like petrol/fuel bills, the model maps data to its known schema (e.g., labeling fuel types as `menu_item` or liters as `count`).
* **Future Work:** To improve accuracy for specific industries (like logistics or petrol bills), the model would benefit from fine-tuning on the **SROIE dataset** or custom-annotated local invoices.
