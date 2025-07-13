#!/bin/bash
echo "Setting up spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
echo "Setup completed!" 