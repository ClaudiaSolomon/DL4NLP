import requests
import re
from collections import defaultdict, Counter

def fetch_romanian_corpus(min_words=1000):
	url = "https://info.uaic.ro/intrebari-frecvente-studenti"
	response = requests.get(url)
	text = response.text
	text = re.sub(r"<.*?>", " ", text)

	text = re.sub(r"[^a-zA-ZăîâșțĂÎÂȘȚ ]", " ", text)
	words = text.lower().split()
	words = [w for w in words if len(w) > 1]
	while len(words) < min_words:
		words += words
	return words[:min_words]

class NGramLM:
	def __init__(self, n):
		self.n = n
		self.ngram_counts = defaultdict(int)
		self.context_counts = defaultdict(int)
		self.vocab = set()

	def train(self, corpus):
		self.vocab = set(corpus)
		padded = ["<s>"] * (self.n - 1) + corpus + ["</s>"]
		for i in range(len(padded) - self.n + 1):
			ngram = tuple(padded[i:i+self.n])
			context = tuple(padded[i:i+self.n-1])
			self.ngram_counts[ngram] += 1
			self.context_counts[context] += 1

	def prob(self, ngram):
		context = ngram[:-1]
		V = len(self.vocab)
		return (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + V)

	def generate(self, max_len=20):
		result = ["<s>"] * (self.n - 1)
		for _ in range(max_len):
			context = tuple(result[-(self.n-1):])
			candidates = [(ngram[-1], self.prob(ngram)) for ngram in self.ngram_counts if ngram[:-1] == context]
			if not candidates:
				break
			next_word = max(candidates, key=lambda x: x[1])[0]
			if next_word == "</s>":
				break
			result.append(next_word)
		return " ".join(result[self.n-1:])

	def sentence_prob(self, sentence):
		words = sentence.lower().split()
		padded = ["<s>"] * (self.n - 1) + words + ["</s>"]
		prob = 1.0
		for i in range(len(padded) - self.n + 1):
			ngram = tuple(padded[i:i+self.n])
			p = self.prob(ngram)
			prob *= p
		return prob

if __name__ == "__main__":
	print("Fetching Romanian corpus...")
	corpus = fetch_romanian_corpus(1000)
	print(f"Corpus size: {len(corpus)} words")
	n = 3
	lm = NGramLM(n)
	lm.train(corpus)
	print(f"Trigram probability example:")
	example_ngram = tuple(["<s>"] * (n-1) + [corpus[0]])
	print(f"P({example_ngram}) = {lm.prob(example_ngram):.6f}")
	print("\nGenerated text:")
	print(lm.generate(10))

	new_sentence = input("\nEnter a Romanian sentence to compute its probability: ")
	p = lm.sentence_prob(new_sentence)
	print(f"Probability of the sentence: {p:.10f}")
