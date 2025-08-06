import spacy
import nltk
from nltk.stem import PorterStemmer



# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

text = """
Hello I am Atharva Bhavar is pursuing B.Tech in Information Technology at Sanjivani College of Engineering.
I am Living in Kopargaon, Maharashtra, India.
I am passionate about solving problems using technology.
"""

# Process the text
doc = nlp(text)

# Sentence Tokenization
sentences = [sent.text.strip() for sent in doc.sents]
print("\n Sentences:", sentences)

# Word Tokenization (excluding punctuation)
words = [token.text for token in doc if not token.is_punct]
print("\n Words:", words)

# Filtered words (removing stopwords and punctuation)
filtered_words = [token.text for token in doc if not token.is_punct and not token.is_stop]
print("\n Filtered Words:", filtered_words)

# Lemmatization
lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
print("\n Lemmatized Words:", lemmatized_words)

# Stemming (PorterStemmer)
stemmed_words = [stemmer.stem(token.text) for token in doc if not token.is_punct and not token.is_stop]
print("\n Stemmed Words:", stemmed_words)
