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

# Sidebar navigation
with st.sidebar:
    st.title("🧭 Navigation")
    page = st.radio("Choose a page:", ["🤖 NER Inference", "📚 Dataset Documentation"])
    
    st.markdown("---")
    st.markdown("**🔗 Quick Links:**")
    st.markdown("• [Dataset on Kaggle](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)")
    st.markdown("• [GitHub Repository](https://github.com/MON3EMPASHA/NER-for-New-Articles-trained-on-CoNLL03-Dataset.git)")
    st.markdown("• [My Portfolio](https://abdelmonem-hatem.netlify.app/)")

# Page 1: NER Inference
if page == "🤖 NER Inference":
    st.markdown("""
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

# Page 2: Dataset Documentation
elif page == "📚 Dataset Documentation":
    st.title("📚 CoNLL-2003 Dataset Documentation")
    
    st.markdown("""
    
    #### 📌 Overview
    The CoNLL-2003 dataset is a benchmark dataset for training and evaluating Named Entity Recognition (NER) systems. It was introduced in the CoNLL-2003 Shared Task, focused on language-independent NER.
    
    It contains annotated text from Reuters news stories, originally collected for research in NLP.
    
    #### 🌍 Supported Languages
    • English 🇬🇧
    • German 🇩🇪
    
    Most popular usage is with English, so examples here will focus on that.
    
    #### 🎯 Task Objective
    Identify named entities in raw text and classify them into 4 types:
    
    | Entity Type | Description | Example |
    |-------------|-------------|---------|
    | PER | Person | Barack Obama |
    | ORG | Organization | Microsoft |
    | LOC | Location | Egypt |
    | MISC | Miscellaneous (e.g., nationalities, events) | Egyptian, Olympics |
    
    #### 🏷️ Tagging Format
    The dataset uses the IOB (Inside-Outside-Beginning) scheme:
    
    | Tag | Meaning |
    |-----|---------|
    | B-XXX | Beginning of entity type XXX |
    | I-XXX | Inside (continuation) of entity |
    | O | Outside of any named entity |
    
    **Example:**
    ```
    John     B-PER
    Smith    I-PER
    works    O
    at       O
    Google   B-ORG
    .        O
    ```
    
    #### 📑 Dataset File Format
    The dataset is usually presented in token-level annotated format, where each line represents one word/token, along with annotations.
    
    Each line contains 4 columns:
    
    | Column Index | Description | Example |
    |--------------|-------------|---------|
    | 1 | Word/token | London |
    | 2 | POS tag (Part-of-speech) | NNP |
    | 3 | Chunk tag (phrase structure) | I-NP |
    | 4 | NER tag | B-LOC |
    
    Sentences are separated by empty lines.
    
    #### 📦 Sample (English)
    ```
    U.N.     NNP     I-NP     B-ORG
    official NNP     I-NP     O
    Ekeus    NNP     I-NP     B-PER
    heads    VBZ     I-VP     O
    for      IN      I-PP     O
    Baghdad  NNP     I-NP     B-LOC
    .        .       O        O
    ```
    
    **Explanation:**
    • U.N. is an organization → B-ORG
    • Ekeus is a person → B-PER
    • Baghdad is a location → B-LOC
    • official, heads, for, . → not part of named entities → O
    
    #### 📁 Files in the Dataset
    Typically split into:
    • train.txt → Training set
    • valid.txt (or dev.txt) → Development/Validation set
    • test.txt → Test set
    
    #### 🧠 Column Tags Breakdown
    
    **1. Word/Token**
    The actual word from the sentence.
    
    **2. POS Tag (Penn Treebank)**
    Examples:
    
    | Tag | Description |
    |-----|-------------|
    | NNP | Proper noun, singular |
    | VBZ | Verb, 3rd person singular |
    | DT | Determiner |
    | IN | Preposition |
    | JJ | Adjective |
    
    **3. Chunk Tag**
    Syntactic grouping of words into phrases:
    
    | Tag | Meaning |
    |-----|---------|
    | B-NP | Beginning of Noun Phrase |
    | I-NP | Inside Noun Phrase |
    | B-VP | Beginning of Verb Phrase |
    | O | Outside any phrase |
    
    These are useful for shallow parsing.
    
    **4. NER Tag**
    NER annotation in BIO format (described earlier).
    
    ---
    
    **📖 Dataset Source:** [CoNLL-2003 Dataset on Kaggle](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)
    """) 