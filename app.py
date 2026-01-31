import torch
import os
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import pandas as pd

def initialize_model():
    print("🔄 Loading AI Model...")
    model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def extract_data(image_path, processor, model, device):
    """
    Performs OCR-free document understanding.
    """
    # Load and prepare image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Prepare the task prompt for the model
    # <s_cord-v2> is the special token for receipt parsing
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Generate output
    print("AI is analyzing the receipt...")
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Convert output tokens to JSON
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task token
    
    return processor.token2json(sequence)

def save_to_tracker(extracted_data, filename="receipt_log.xlsx"):
    # Convert the JSON dictionary to a clean list/table
    # Note: Depending on your receipt, you might need to flatten the JSON
    df = pd.json_normalize(extracted_data) 
    
    # If the file exists, append; otherwise, create new
    if not os.path.isfile(filename):
        df.to_excel(filename, index=False)
    else:
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
             df.to_excel(writer, index=False)
    print(f" Data saved to {filename}")

if __name__ == "__main__":
    proc, mod, dev = initialize_model()
    
    # Check for images in data/
    data_folder = "data"
    images = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if images:
        path = os.path.join(data_folder, images[0])
        result = extract_data(path, proc, mod, dev)
        
        print("\n--- EXTRACTED DATA ---")
        print(result)
        print("------------------------\n")
        save_to_tracker(result) 
    else:
        print("No images found in 'data/' folder.")