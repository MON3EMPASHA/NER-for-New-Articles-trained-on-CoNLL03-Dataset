# News Article NER Streamlit App

This application provides an interactive web interface for Named Entity Recognition (NER) on news articles using spaCy models. You can input your own text or select from example news snippets, and the app will highlight and list recognized entities such as Person, Organisation, Location, Date, and Miscellaneous.

## Features
- **NER with spaCy**: Uses both small and large English models trained on the CoNLL-2003 dataset.
- **Entity Visualization**: Highlights entities in the text with color-coded labels.
- **Entity Table**: Lists all detected entities with user-friendly labels.
- **Model Comparison**: Run either or both models and compare results.
- **Example Texts**: Quickly test the app with preloaded news article examples.

## Setup
1. **Clone the repository or copy the files to your project directory.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download or ensure the following spaCy models are available in `saved_models/`:**
   - `en_core_web_sm`
   - `en_core_web_lg`
   (These should be exported using `spacy`'s `to_disk` method as shown in the notebook.)

## Usage
Run the Streamlit app with:
```bash
streamlit run app.py
```

- Enter or paste your news article text, or select an example from the dropdown.
- Choose which model(s) to use.
- Click **Run NER** to see highlighted entities and a table of results.

## File Structure
- `app.py` — Main Streamlit application.
- `requirements.txt` — Python dependencies.
- `saved_models/` — Directory containing exported spaCy models.
- `README.md` — This file.

## Credits
- Built with [spaCy](https://spacy.io/) and [Streamlit](https://streamlit.io/).
- Models trained on the [CoNLL-2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/).

---
For questions or improvements, feel free to open an issue or contact the author. 