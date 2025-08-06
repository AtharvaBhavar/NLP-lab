import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Input text
example_string = """
Hi I am Atharva Bhavar is pursuing B.Tech in Information Technology at Sanjivani College of Engineering.
I am Living in Kopargaon, Maharashtra, India.
I am passionate about solving problems using technology.
"""

# Sentence Tokenization
sentences = sent_tokenize(example_string)
print("Tokenized Sentences:\n", sentences)

# Word Tokenization
words = word_tokenize(example_string)

# Remove punctuation/symbols
words = [word for word in words if word.isalnum()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# Output results
print("\n Filtered Words (No stopwords/punctuation):\n", filtered_words)
print("\n Stemmed Words:\n", stemmed_words)
print("\n Lemmatized Words:\n", lemmatized_words)

