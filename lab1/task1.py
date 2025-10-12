import numpy as np
from collections import Counter
import math

def preprocess_sentence(sentence):
    return sentence.lower().split()

S1 = "The man saw a car in the park"
S2 = "I saw the man park the car"


if __name__ == "__main__":
    words_S1 = preprocess_sentence(S1)
    words_S2 = preprocess_sentence(S2)

    # Euclidean
    vocab = sorted(list(set(words_S1 + words_S2)))
    vector_S1_binary = [1 if word in words_S1 else 0 for word in vocab]
    vector_S2_binary = [1 if word in words_S2 else 0 for word in vocab]
    euclidean_distance = np.sqrt(sum((a - b)**2 for a, b in zip(vector_S1_binary, vector_S2_binary)))
    euclidean_similarity = 1 / (1 + euclidean_distance)
    print("a) Euclidean")
    print("Vector representation:")
    print("S1:", vector_S1_binary)
    print("S2:", vector_S2_binary)
    print("Similarity:", euclidean_similarity)

    # Vector cosine
    freq_S1 = Counter(words_S1)
    freq_S2 = Counter(words_S2)
    vector_S1_freq = [freq_S1[word] for word in vocab]
    vector_S2_freq = [freq_S2[word] for word in vocab]
    dot_product = sum(a * b for a, b in zip(vector_S1_freq, vector_S2_freq))
    magnitude_S1 = math.sqrt(sum(a**2 for a in vector_S1_freq))
    magnitude_S2 = math.sqrt(sum(a**2 for a in vector_S2_freq))
    cosine_similarity = dot_product / (magnitude_S1 * magnitude_S2) if magnitude_S1 * magnitude_S2 != 0 else 0
    print("b) Vector cosine")
    print("Vector representation:")
    print("S1:", vector_S1_freq)
    print("S2:", vector_S2_freq)
    print("Similarity:", cosine_similarity)

    # Jaccard
    set_S1 = set(words_S1)
    set_S2 = set(words_S2)
    intersection = set_S1.intersection(set_S2)
    union = set_S1.union(set_S2)
    jaccard_similarity = len(intersection) / len(union) if len(union) != 0 else 0
    print("c) Jaccard")
    print("Set representation:")
    print("S1:", set_S1)
    print("S2:", set_S2)
    print("Similarity:", jaccard_similarity)

    # Overlap
    min_size = min(len(set_S1), len(set_S2))
    overlap_similarity = len(intersection) / min_size if min_size != 0 else 0
    print("d) Overlap")
    print("Set representation:")
    print("S1:", set_S1)
    print("S2:", set_S2)
    print("Similarity:", overlap_similarity)