import nltk
from nltk import CFG

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | Det Adj N | Det N PP | NP Conj NP | Adj N | V N
VP -> V | V NP | V NP PP | V Adv | V PP | Aux VP | V Comp NP | Aux V Adj | V NP Adv Comp NP
PP -> P NP
Adj -> "dangerous" | "flying"
Adv -> "more"
Comp -> "than"
Conj -> "and"
Det -> "the"
N -> "planes" | "parents" | "bride" | "groom"
V -> "flying" | "loves" | "be"
Aux -> "can" | "were"
P -> "of"
""")


parser = nltk.ChartParser(grammar)

sentences = [
    "flying planes can be dangerous",
    "the parents of the bride and the groom were flying",
    "the groom loves dangerous planes more than the bride"
]

for sent in sentences:
    print(f"\nSentence: {sent}")
    for tree in parser.parse(sent.split()):
        print(tree)
        tree.pretty_print()