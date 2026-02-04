import streamlit as st
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import pandas as pd
import io

# --- 1. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    """Loads the model once and shares it across all sessions to save memory."""
    model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

# --- 2. EXTRACTION LOGIC ---
def extract_data(image, processor, model, device):
    """Processes image through Swin-BART architecture to get structured JSON."""
    pixel_values = processor(image, return_tensors="pt").pixel_values
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return processor.token2json(sequence)

@st.cache_data
def convert_df(df_to_convert):
    """Caches the CSV conversion so it doesn't rerun on every click."""
    return df_to_convert.to_csv(index=False).encode('utf-8')

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Donut Receipt Tracker", page_icon="🍩")
st.title("🍩 Receipt-AI Tracker")
st.markdown("Automate your expense logging using **OCR-free Transformers**.")

# Load resources
processor, model, device = load_model()

# Sidebar for Technical Context
with st.sidebar:
    st.header("Technical Overview")
    st.info("Architecture: **Swin-BART** (Donut)")
    st.write("This model processes raw pixels directly into JSON, bypassing traditional OCR.")
    if st.checkbox("Show Training Data Info"):
        st.warning("**Domain Bias Note:** Trained on the CORD dataset (restaurant/retail). May struggle with petrol or utility bills.")

# File Uploader
uploaded_file = st.file_uploader("Upload a receipt (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Receipt Preview", use_container_width=True)
    
    # Process Button
    if st.button("Extract & Structure Data"):
        with st.spinner("AI is analyzing document layout..."):
            # Extraction
            result = extract_data(image, processor, model, device)
            
            # Tabular Preview
            st.subheader("📊 Itemized Breakdown")
            
            try:
                # Check if 'menu' exists and is a list
                if 'menu' in result and isinstance(result['menu'], list):
                    # record_path flattens the list into rows
                    # meta keeps the 'total' or 'date' attached to every row
                    meta_keys = [k for k in result.keys() if k != 'menu']
                    df = pd.json_normalize(result, record_path=['menu'], meta=meta_keys, errors='ignore')
                else:
                    df = pd.json_normalize(result)
                
                # Sanitize for Streamlit display
                st.dataframe(df.astype(str))
                
            except Exception as e:
                st.warning("Could not itemize rows automatically. Showing raw table instead.")
                st.dataframe(pd.json_normalize(result).astype(str))
            
            # Download Section
            csv_data = convert_df(df)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="receipt_data.csv",
                mime="text/csv",
            )
            
            # Raw JSON for transparency
            with st.expander("View Raw JSON Output"):
                st.json(result)