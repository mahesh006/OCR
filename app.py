import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import tempfile
import easyocr
import re


# Load the pre-trained English OCR model and tokenizer 
@st.cache_resource
def load_english_model():
    tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval()
    return tokenizer, model

# Load the multilingual OCR model (EasyOCR) for both Hindi and English
@st.cache_resource
def load_multilingual_model():
    reader = easyocr.Reader(['en', 'hi'])  
    return reader

st.title('OCR Web App')

if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = ""

# Function to highlight the search word in the OCR results
def highlight_text(text, search_query):
    if not search_query:
        return text
    highlighted = re.sub(f"({re.escape(search_query)})", r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return highlighted

# Create a two-column layout
left_column, right_column = st.columns([1, 2])

with left_column:
    language = st.selectbox("Select Language", ["English", "Hindi + English"])

    predict_button = st.button('Predict')

    # Search functionality after results
    search_query = st.text_input("Search in results")
    search_button = st.button('Search')

    # Display search results by highlighting the searched word
    if search_button and st.session_state['ocr_result']:
        if search_query.lower() in st.session_state['ocr_result'].lower():
            st.success(f"'{search_query}' found in the OCR results!")
        else:
            st.error(f"'{search_query}' not found in the OCR results.")

with right_column:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_format = image.format if image.format is not None else "JPEG"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img_format.lower()}") as tmp_file:
            img_path = tmp_file.name
            image.save(img_path, format=img_format)

        # Perform OCR when the Predict button is clicked
        if predict_button:
            if language == "English":
                tokenizer, model = load_english_model()
                with st.spinner('Processing English OCR...'):
                    res = model.chat(tokenizer, img_path, ocr_type='ocr')
                st.session_state['ocr_result'] = res  
                st.write("OCR Result (English):")
                st.write(st.session_state['ocr_result'], unsafe_allow_html=True)

            elif language == "Hindi + English":
                reader = load_multilingual_model()
                with st.spinner('Processing Hindi + English OCR...'):
                    result = reader.readtext(img_path, detail=0)  
                st.session_state['ocr_result'] = " ".join(result)  
                st.write("OCR Result (Hindi + English):")
                st.write(st.session_state['ocr_result'], unsafe_allow_html=True)

    if st.session_state['ocr_result']:
        highlighted_result = highlight_text(st.session_state['ocr_result'], search_query)
        st.write("Highlighted OCR Result:")
        st.markdown(highlighted_result, unsafe_allow_html=True)