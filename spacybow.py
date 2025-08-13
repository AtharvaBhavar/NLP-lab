import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

# Process text
doc = nlp(paragraph)

corpus = []

# Sentence-wise lemmatization & stopword removal
for sent in doc.sents:
    tokens = [
        token.lemma_.lower()
        for token in sent
        if token.is_alpha and not token.is_stop
    ]
    corpus.append(" ".join(tokens))

print("Corpus:", corpus)

# Tokenized unique words per sentence
words_unique = [sent.split() for sent in corpus]
print("Words Unique:", words_unique)

# Bag of Words
cv = CountVectorizer()
independentFeatures_bow = cv.fit_transform(corpus).toarray()
print("BOW:", independentFeatures_bow)

# TF-IDF
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()
print("TF-IDF:", independentFeatures_tfIDF)
