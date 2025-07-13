import streamlit as st
import spacy
from spacy import displacy
import os
import tempfile
from pathlib import Path
import streamlit.components.v1 as components
import sys

# Set page config
st.set_page_config(page_title="News Article NER Demo", layout="wide")

st.markdown("""
**About this app:**

This application demonstrates Named Entity Recognition (NER) for news articles using two spaCy models (small and large) trained on the CoNLL-2003 dataset. NER is the process of identifying and classifying key information (entities) in text, such as people, organizations, and locations. 

- **Input:** Paste or type a news article or sentence.
- **Output:** The app highlights and lists recognized entities of types: PERSON, ORG, LOC, Date and MISC.
- **Models:** You can compare the results of the small and large spaCy models.

This tool helps you explore how well the models extract entities from real-world news text.
""")

st.title("📰 Named Entity Recognition for News Articles")
st.write("Test the spaCy NER models (small and large) trained on news data. Enter your own text and see the recognized entities, both highlighted and listed.")

# Model loading (cache for performance)
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        nlp_sm = spacy.load("en_core_web_sm")
        nlp_lg = spacy.load("en_core_web_lg")
        return nlp_sm, nlp_lg
    except OSError as e:
        st.error(f"""
        **Error: spaCy models not found!**
        
        The required spaCy models are not available in this environment.
        Error: {str(e)}
        
        **For local development:**
        Please install the required spaCy models by running these commands in your terminal:
        ```
        python -m spacy download en_core_web_sm
        python -m spacy download en_core_web_lg
        ```
        
        **For Streamlit Cloud deployment:**
        The models need to be included in the deployment package. Please check the deployment configuration.
        """)
        st.stop()

try:
    nlp_sm, nlp_lg = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Model selection
model_options = {
    "Small (en_core_web_sm)": nlp_sm,
    "Large (en_core_web_lg)": nlp_lg,
    "Both": None
}
model_choice = st.radio("Select model(s) to use:", list(model_options.keys()), index=2)

# Example texts
examples = [
    "Custom input",
    "The U.N. official Ekeus warned of a potential conflict in Iraq if Baghdad refuses weapons inspections.",
    "Germany's representative at the U.N., Hans Schmidt, will visit New York next week.",
    "Apple unveiled a new iPhone model in September, aiming to compete with Samsung.",
    "Prime Minister Tony Blair met with Microsoft executives in London to discuss technology investments.",
    "The European Union imposed sanctions on Russia following the annexation of Crimea."
]

example_choice = st.selectbox("Choose an example or select 'Custom input':", examples)

if example_choice != "Custom input":
    user_text = st.text_area("Enter news article text:", value=example_choice, height=150, key="text_area")
else:
    user_text = st.text_area("Enter news article text:", height=150, placeholder="Paste or type your news article here...", key="text_area")

run_button = st.button("Run NER")

if run_button and user_text.strip():
    models_to_run = []
    if model_choice == "Both":
        models_to_run = [("Small (en_core_web_sm)", nlp_sm), ("Large (en_core_web_lg)", nlp_lg)]
    else:
        models_to_run = [(model_choice, model_options[model_choice])]

    for label, nlp in models_to_run:
        st.subheader(f"Results: {label}")
        doc = nlp(user_text)
        # Visualize entities using displacy (render as HTML)
        html = displacy.render(doc, style="ent", minify=True, jupyter=False)
        html = html.replace(">GPE<", ">LOC<")  # Replace GPE with LOC in visualization
        # Streamlit safe HTML embedding
        components.html(f"""
        <div style='padding:1em;border:1px solid #eee;border-radius:8px;background:#fafafa'>
        {html}
        </div>
        """, height=150 + 30 * (user_text.count("\n") + 1))

        # List entities
        if doc.ents:
            st.markdown("**Entities found:**")
            def map_label(label):
                if label in ["LOC"]:
                    return "Location"
                elif label in ["PER", "PERSON"]:
                    return "Person"
                elif label == "ORG":
                    return "Organisation"
                else:
                    return label
            ent_data = [(ent.text, map_label("LOC" if ent.label_ == "GPE" else ent.label_)) for ent in doc.ents]
            st.table(ent_data)
        else:
            st.info("No entities found.")
elif run_button and not user_text.strip():
    st.warning("Please enter some text before running the model.")
else:
    st.info("Enter some text above and press 'Run NER' to see results.") 