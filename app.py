import streamlit as st
import easyocr
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy as np
import os
from pathlib import Path
from gliner import GLiNER

# Set environment variables for model storage
os.environ['GLINER_HOME'] = str(Path.home() / '.gliner_models')
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.gliner_models' / 'cache')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def get_model_path():
    """Get the path to the local model directory."""
    base_dir = Path.home() / '.gliner_models'
    model_dir = base_dir / 'gliner_large-v2.1'
    return model_dir

def download_model():
    """Download the model if it doesn't exist locally."""
    model_dir = get_model_path()
    if not model_dir.exists():
        st.info("Downloading GLiNER model for the first time... This may take a few minutes.")
        try:
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
            temp_model.save_pretrained(str(model_dir))
            st.success("Model downloaded successfully!")
            return temp_model
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            raise e
    return None

@st.cache_resource
def load_gliner_model():
    """Load the GLiNER model, downloading it if necessary."""
    model_dir = get_model_path()
    if model_dir.exists():
        try:
            return GLiNER.from_pretrained(str(model_dir))
        except Exception as e:
            st.warning("Error loading existing model. Attempting to redownload...")
            import shutil
            shutil.rmtree(model_dir, ignore_errors=True)
    
    model = download_model()
    if model:
        return model
    return GLiNER.from_pretrained(str(model_dir))

def extract_text_from_image(image):
    """Extracts text from a single image using EasyOCR."""
    image_array = np.array(image)
    return reader.readtext(image_array, detail=0, paragraph=True)

def process_entities(text: str, model, threshold: float, nested_ner: bool) -> dict:
    """Process text with GLiNER model - matching app.py implementation."""
    # Define our business card labels
    labels = "person name, company name, job title, phone, email, address"
    labels = [label.strip() for label in labels.split(",")]
    
    # Get predictions
    entities = model.predict_entities(
        text, 
        labels, 
        flat_ner=not nested_ner,
        threshold=threshold
    )
    
    # Format results matching app.py structure
    formatted_entities = []
    for entity in entities:
        formatted_entities.append({
            "entity": entity["label"],
            "word": entity["text"],
            "start": entity["start"],
            "end": entity["end"]
        })
    
    # Organize results by category
    results = {
        "Person Name": [],
        "Company Name": [],
        "Job Title": [],
        "Phone": [],
        "Email": [],
        "Address": []
    }
    
    for entity in formatted_entities:
        category = entity["entity"].title()
        if category in results:
            results[category].append(entity["word"])
    
    # Join multiple entries with semicolons
    return {k: "; ".join(set(v)) if v else "" for k, v in results.items()}

def main():
    st.title("Business Card Information Extractor")
    
    # Model settings in sidebar
    st.sidebar.title("Settings")
    
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values will detect more entities (as in app.py example)"
    )
    
    nested_ner = st.sidebar.checkbox(
        "Enable Nested NER",
        value=True,
        help="Allow detection of nested entities"
    )

    # Upload options
    upload_type = st.sidebar.radio("Upload Type", ("Single", "Batch"))
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Business Card Image(s)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=(upload_type == "Batch")
    )

    if uploaded_files:
        # Load model
        model = load_gliner_model()
        
        # Process files
        results = []
        files_to_process = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        
        progress_bar = st.progress(0)
        for idx, file in enumerate(files_to_process):
            with st.expander(f"Processing {file.name}"):
                # Load and extract text
                image = Image.open(file)
                extracted_text = extract_text_from_image(image)
                clean_text = " ".join(extracted_text)
                
                # Show extracted text
                st.text("Extracted Text:")
                st.text(clean_text)
                
                # Process with GLiNER
                result = process_entities(clean_text, model, threshold, nested_ner)
                result["File Name"] = file.name
                results.append(result)
                
                # Show individual results
                st.json(result)
            
            progress_bar.progress((idx + 1) / len(files_to_process))
        
        # Show final results
        if results:
            st.success("Processing Complete!")
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns to put filename first
            cols = ["File Name"] + [col for col in df.columns if col != "File Name"]
            df = df[cols]
            
            # Display results
            st.dataframe(df, use_container_width=True)
            
            # Provide download option
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                "business_card_results.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()