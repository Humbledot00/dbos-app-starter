from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import spacy
from typing import List, Dict
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from processing import generate_mnemonic, generate_story, generate_summary


app = Flask(__name__)
CORS(app) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################
# TEXT ANALYSIS (Key Points Extraction)
########################################################################################
class TextAnalyzer:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        text = re.sub(r'[^\w\s.]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return text.strip()

    def get_key_points(self, text: str) -> Dict[str, List[str]]:
        """Extract key points from the input text using NLP techniques."""
        cleaned_text = self.preprocess_text(text)
        doc = self.nlp(cleaned_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Calculate sentence scores
        sentence_scores = []
        for i, sent in enumerate(doc.sents):
            score = np.sum(tfidf_matrix[i].toarray())
            entities_bonus = len([ent for ent in sent.ents])
            pos_bonus = len([token for token in sent if token.pos_ in ['NOUN', 'VERB', 'PROPN']])
            total_score = score + (0.1 * entities_bonus) + (0.05 * pos_bonus)
            sentence_scores.append((sent.text.strip(), total_score))
        
        ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        entities = [ent.text for ent in doc.ents]
        key_entities = Counter(entities).most_common(5)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        main_topics = Counter(noun_chunks).most_common(5)
        
        return {
            'key_points': [sent for sent, score in ranked_sentences[:3]],
            'main_topics': [topic for topic, count in main_topics],
        }


def process_text(sample_text):
    analyzer = TextAnalyzer()
    simplifier = TextSimplifier()

    # Extract key points and topics
    result = analyzer.get_key_points(sample_text)
    key_points = result['key_points']
    main_topics = result['main_topics']

    # Generate summaries and mnemonic
    simplified_text = simplifier.simplify_text(sample_text)
    mnemonic = generate_mnemonic(sample_text)
    summary = generate_summary(sample_text)
    story = generate_story(sample_text)

    # Return a JSON response
    response_json = {
        "story": story,
        "key_points": key_points,
        "main_topics": main_topics,
        "simplified_text": simplified_text,
        "mnemonic": mnemonic,
        "summary": summary
    }

    return response_json



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    print(data)
    input_text = data.get('input_text', '')
    processed_html = "process_text(input_text)"
    return jsonify({'generated_sentence': processed_html})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
