import wikipedia
import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD, NMF
import numpy as np
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

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
    print(
        f"Transformed shape: {svd_model.transform(bow_matrix).shape if name == 'BoW' else svd_model.transform(tfidf_matrix).shape}")
    print(f"Explained variance ratio (sum): {explained:.4f}")
    top_n = 10
    for i, comp in enumerate(svd_model.components_):
        top_idx = np.argsort(comp)[::-1][:top_n]
        top_terms = [terms[idx] for idx in top_idx]
        print(f"Component {i + 1}: {', '.join(top_terms)}")


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
        print(f"Component {i + 1}: {', '.join(top_terms)}")


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
print("Done: LSA and NMF computed successfully.")

print("\n=== Latent Dirichlet Allocation (LDA) ===")

texts = [doc.split() for doc in processed_docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=n_components,
    random_state=42,
    passes=20,
    alpha='auto',
    eta='auto'
)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx + 1}: {topic}")

coherence_model_lda = CoherenceModel(
    model=lda_model,
    texts=texts,
    dictionary=dictionary,
    coherence='c_v',
    processes=1
)
coherence_lda = coherence_model_lda.get_coherence()
print(f"\nCoherence Score (LDA): {coherence_lda:.4f}")

perplexity = lda_model.log_perplexity(corpus)
print(f"Perplexity (LDA): {perplexity:.4f}")

LDAvis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(LDAvis_data, 'lda_topics_visualization.html')
print("LDA visualization saved as 'lda_topics_visualization.html'")

print("\nDone: LDA and evaluation metrics computed successfully.")
