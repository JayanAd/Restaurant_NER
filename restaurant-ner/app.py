import streamlit as st
import pandas as pd
from transformers import pipeline

# Load the NER model from Hugging Face
ner_pipeline = pipeline("token-classification", model="JynPortfolio/restaurant-ner-distilbert", grouped_entities=True)

st.title("Restaurant Named Entity Recognition (NER) Platform")

st.write("This platform allows you to input restaurant-related text and extracts entities using a finetuned DistilBERT by Jayan Adhikari on MIT Restaurant dataset.")

# Input section
user_input = st.text_area("Enter text to analyze for entities:", placeholder="Type your restaurant-related text here...")

# Predict NER entities when the user provides input
if st.button("Analyze Text"):
    if user_input:
        # Perform NER using the pipeline
        ner_results = ner_pipeline(user_input)

        # Process the results for display
        entities = [{"Entity": result['word'], "Label": result['entity_group'],"Score":result["score"]} for result in ner_results]

        st.write("### Recognized Entities")
        
        # Display results in a table if entities are found
        if entities:
            entity_df = pd.DataFrame(entities)
            st.table(entity_df)
        else:
            st.write("No entities recognized.")
    else:
        st.warning("Please enter some text to analyze.")