import spacy

sentences = [
    "Flying planes can be dangerous.",
    "The parents of the bride and the groom were flying.",
    "The groom loves dangerous planes more than the bride."
]

MODEL = "en_core_web_sm"
nlp = spacy.load(MODEL)

for i, sent in enumerate(sentences, 1):
    doc = nlp(sent)
    header = f"Sentence {i}: {sent.strip()}"
    print(header)

    print(f"{'Token':15}{'Dep':10}{'Head':15}{'HeadPos':10}")
    for tok in doc:
        line = f"{tok.text:15}{tok.dep_:10}{tok.head.text:15}{tok.head.pos_:10}"
        print(line)

    triples = [(tok.text, tok.dep_, tok.head.text) for tok in doc]
    print("\nDependency triples:")
    print(triples)

    print("\n" + ("-" * 60) + "\n")
