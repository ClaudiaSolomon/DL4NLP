import wikipedia
import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD, NMF
import numpy as np

topics = {
"plants": ["Lavender", "Cactus", "Rose", "Sunflower", "Chrysanthemum"],
"cakes": ["Biscuit", "Muffin", "Croissant", "Cheesecake", "Molten chocolate cake"],
"fish": ["Siamese fighting fish", "Pufferfish", "Anglerfish", "Clownfish", "Octopus"]
}

documents = []
titles = []

for category, pages in topics.items():
    for page in pages:
        try:
            summary = wikipedia.page(page).summary
            documents.append(summary)
            titles.append(page)
        except Exception as e:
            print(f"Could not fetch page '{page}': {e}")


stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[\d\W_]+', ' ', text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    stems = [stemmer.stem(t) for t in tokens]
    return ' '.join(stems)

processed_docs = [preprocess(t) for t in documents]

bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

bow_matrix = bow_vectorizer.fit_transform(processed_docs)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)


df = pd.DataFrame({
"Title": titles,
"Original (first 300 chars)": [d[:300] + "..." for d in documents],
"Preprocessed": processed_docs
})

print("=== Original vs Preprocessed Texts ===")
print(df, "\n")

print("Vocabulary size (BoW):", len(bow_vectorizer.get_feature_names_out()))
print("First 10 words in vocabulary:", bow_vectorizer.get_feature_names_out()[:10], "\n")

n_components = 5
print(f"\n=== Latent Semantic Analysis (LSA) with {n_components} components ===")

def print_lsa_results(svd_model, vectorizer, name):
    terms = vectorizer.get_feature_names_out()
    explained = svd_model.explained_variance_ratio_.sum()
    print(f"\n-- {name} LSA --")
    print(f"Transformed shape: {svd_model.transform(bow_matrix).shape if name=='BoW' else svd_model.transform(tfidf_matrix).shape}")
    print(f"Explained variance ratio (sum): {explained:.4f}")
    top_n = 10
    for i, comp in enumerate(svd_model.components_):
        top_idx = np.argsort(comp)[::-1][:top_n]
        top_terms = [terms[idx] for idx in top_idx]
        print(f"Component {i+1}: {', '.join(top_terms)}")

svd_bow = TruncatedSVD(n_components=n_components, random_state=42)
bow_lsa = svd_bow.fit_transform(bow_matrix)
print_lsa_results(svd_bow, bow_vectorizer, 'BoW')

svd_tfidf = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_lsa = svd_tfidf.fit_transform(tfidf_matrix)
print_lsa_results(svd_tfidf, tfidf_vectorizer, 'TF-IDF')

print(f"\n=== Non-negative Matrix Factorization (NMF) with {n_components} components ===")

def print_nmf_results(nmf_model, components, vectorizer, name):
    terms = vectorizer.get_feature_names_out()
    print(f"\n-- {name} NMF --")
    print(f"Components shape: {components.shape}")
    print(f"Reconstruction error: {getattr(nmf_model, 'reconstruction_err_', None)}")
    top_n = 10
    for i, comp in enumerate(components):
        top_idx = np.argsort(comp)[::-1][:top_n]
        top_terms = [terms[idx] for idx in top_idx]
        print(f"Component {i+1}: {', '.join(top_terms)}")

nmf_bow = NMF(n_components=n_components, random_state=42, init='nndsvda', max_iter=500)
W_bow = nmf_bow.fit_transform(bow_matrix)
H_bow = nmf_bow.components_
print_nmf_results(nmf_bow, H_bow, bow_vectorizer, 'BoW')

nmf_tfidf = NMF(n_components=n_components, random_state=42, init='nndsvda', max_iter=500)
W_tfidf = nmf_tfidf.fit_transform(tfidf_matrix)
H_tfidf = nmf_tfidf.components_
print_nmf_results(nmf_tfidf, H_tfidf, tfidf_vectorizer, 'TF-IDF')

print("BoW matrix shape:", bow_matrix.shape)
print("TF-IDF matrix shape:", tfidf_matrix.shape)

print("BoW LSA (document x topic) shape:", bow_lsa.shape)
print("TF-IDF LSA (document x topic) shape:", tfidf_lsa.shape)
print("BoW NMF (document x topic) shape:", W_bow.shape)
print("TF-IDF NMF (document x topic) shape:", W_tfidf.shape)

print("Done: Texts processed and vectorized successfully.")
