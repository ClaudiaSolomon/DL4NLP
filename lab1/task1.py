import numpy as np
from collections import Counter
import math

def preprocess_sentence(sentence):
    """Convert sentence to lowercase and split into words"""
    return sentence.lower().split()

# Given sentences
S1 = "The man saw a car in the park"
S2 = "I saw the man park the car"

print("=== Sentence Similarity Analysis ===")
print(f"S1: {S1}")
print(f"S2: {S2}")
print()

# Preprocess sentences
words_S1 = preprocess_sentence(S1)
words_S2 = preprocess_sentence(S2)

print(f"S1 words: {words_S1}")
print(f"S2 words: {words_S2}")
print()

# Create vocabulary (union of all words)
vocab = sorted(list(set(words_S1 + words_S2)))
print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {len(vocab)}")
print()

# a) Euclidean Distance/Similarity
print("=" * 50)
print("a) EUCLIDEAN DISTANCE/SIMILARITY")
print("=" * 50)

# Create binary vectors (1 if word present, 0 if not)
vector_S1_binary = [1 if word in words_S1 else 0 for word in vocab]
vector_S2_binary = [1 if word in words_S2 else 0 for word in vocab]

print("Binary vector representation:")
print(f"S1: {vector_S1_binary}")
print(f"S2: {vector_S2_binary}")

# Compute Euclidean distance
euclidean_distance = np.sqrt(sum((a - b)**2 for a, b in zip(vector_S1_binary, vector_S2_binary)))
print(f"\nEuclidean Distance = √Σ(xi - yi)²")
print(f"Distance = √{sum((a - b)**2 for a, b in zip(vector_S1_binary, vector_S2_binary))} = {euclidean_distance:.4f}")

# Convert to similarity (1 / (1 + distance))
euclidean_similarity = 1 / (1 + euclidean_distance)
print(f"Euclidean Similarity = 1 / (1 + distance) = 1 / (1 + {euclidean_distance:.4f}) = {euclidean_similarity:.4f}")
print()

# b) Vector Cosine Similarity
print("=" * 50)
print("b) VECTOR COSINE SIMILARITY")
print("=" * 50)

# Create frequency vectors
freq_S1 = Counter(words_S1)
freq_S2 = Counter(words_S2)

vector_S1_freq = [freq_S1[word] for word in vocab]
vector_S2_freq = [freq_S2[word] for word in vocab]

print("Frequency vector representation:")
print(f"S1: {vector_S1_freq}")
print(f"S2: {vector_S2_freq}")

# Compute cosine similarity
dot_product = sum(a * b for a, b in zip(vector_S1_freq, vector_S2_freq))
magnitude_S1 = math.sqrt(sum(a**2 for a in vector_S1_freq))
magnitude_S2 = math.sqrt(sum(a**2 for a in vector_S2_freq))

cosine_similarity = dot_product / (magnitude_S1 * magnitude_S2) if magnitude_S1 * magnitude_S2 != 0 else 0

print(f"\nCosine Similarity = (A · B) / (|A| × |B|)")
print(f"Dot product (A · B) = {dot_product}")
print(f"Magnitude |A| = √{sum(a**2 for a in vector_S1_freq)} = {magnitude_S1:.4f}")
print(f"Magnitude |B| = √{sum(a**2 for a in vector_S2_freq)} = {magnitude_S2:.4f}")
print(f"Cosine Similarity = {dot_product} / ({magnitude_S1:.4f} × {magnitude_S2:.4f}) = {cosine_similarity:.4f}")
print()

# c) Jaccard Similarity
print("=" * 50)
print("c) JACCARD SIMILARITY")
print("=" * 50)

set_S1 = set(words_S1)
set_S2 = set(words_S2)

print("Set representation:")
print(f"S1: {set_S1}")
print(f"S2: {set_S2}")

intersection = set_S1.intersection(set_S2)
union = set_S1.union(set_S2)

jaccard_similarity = len(intersection) / len(union) if len(union) != 0 else 0

print(f"\nIntersection (S1 ∩ S2): {intersection}")
print(f"Union (S1 ∪ S2): {union}")
print(f"Jaccard Similarity = |S1 ∩ S2| / |S1 ∪ S2| = {len(intersection)} / {len(union)} = {jaccard_similarity:.4f}")
print()

# d) Overlap Similarity
print("=" * 50)
print("d) OVERLAP SIMILARITY")
print("=" * 50)

print("Set representation (same as Jaccard):")
print(f"S1: {set_S1}")
print(f"S2: {set_S2}")

min_size = min(len(set_S1), len(set_S2))
overlap_similarity = len(intersection) / min_size if min_size != 0 else 0

print(f"\nIntersection (S1 ∩ S2): {intersection}")
print(f"Minimum set size = min(|S1|, |S2|) = min({len(set_S1)}, {len(set_S2)}) = {min_size}")
print(f"Overlap Similarity = |S1 ∩ S2| / min(|S1|, |S2|) = {len(intersection)} / {min_size} = {overlap_similarity:.4f}")
print()

# Summary
print("=" * 50)
print("SUMMARY OF RESULTS")
print("=" * 50)
print(f"Euclidean Similarity:  {euclidean_similarity:.4f}")
print(f"Cosine Similarity:     {cosine_similarity:.4f}")
print(f"Jaccard Similarity:    {jaccard_similarity:.4f}")
print(f"Overlap Similarity:    {overlap_similarity:.4f}")
print()

# Additional analysis
print("=" * 50)
print("DETAILED ANALYSIS")
print("=" * 50)
print("Common words:", intersection)
print("Words only in S1:", set_S1 - set_S2)
print("Words only in S2:", set_S2 - set_S1)
print()
print("Interpretation:")
print("- Higher similarity values indicate more similar sentences")
print("- Cosine similarity considers word frequency")
print("- Jaccard considers presence/absence relative to total unique words")
print("- Overlap considers presence/absence relative to smaller sentence")
print("- Euclidean was converted from distance to similarity")