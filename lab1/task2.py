import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk import word_tokenize
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Ensure WordNet is downloaded
try:
    wn.synsets('dog')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Try to download information content corpus for similarity
try:
    brown_ic = wordnet_ic.ic('ic-brown.dat')
except LookupError:
    nltk.download('wordnet_ic')
    brown_ic = wordnet_ic.ic('ic-brown.dat')

def get_wordnet_relations(word):
    synsets = wn.synsets(word)
    results = {
        'synonyms': set(),
        'antonyms': set(),
        'hypernyms': set(),
        'hyponyms': set(),
        'meronyms': set(),
        'holonyms': set(),
        'definitions': set()
    }
    for syn in synsets:
        # Synonyms
        for lemma in syn.lemmas():
            results['synonyms'].add(lemma.name())
            # Antonyms
            for ant in lemma.antonyms():
                results['antonyms'].add(ant.name())
        # Hypernyms
        for h in syn.hypernyms():
            for lemma in h.lemmas():
                results['hypernyms'].add(lemma.name())
        # Hyponyms
        for h in syn.hyponyms():
            for lemma in h.lemmas():
                results['hyponyms'].add(lemma.name())
        # Meronyms
        for m in syn.part_meronyms() + syn.substance_meronyms() + syn.member_meronyms():
            for lemma in m.lemmas():
                results['meronyms'].add(lemma.name())
        # Holonyms
        for m in syn.part_holonyms() + syn.substance_holonyms() + syn.member_holonyms():
            for lemma in m.lemmas():
                results['holonyms'].add(lemma.name())
        # Definitions
        results['definitions'].add(syn.definition())
    return results

def get_best_synset(word):
    """Get the most common synset for a word."""
    synsets = wn.synsets(word)
    return synsets[0] if synsets else None

def word_similarity(word1, word2):
    """Compute semantic similarity between two words using WordNet path similarity."""
    syn1 = get_best_synset(word1)
    syn2 = get_best_synset(word2)
    if syn1 and syn2:
        # Try several similarity measures, fallback to path_similarity
        sim = syn1.wup_similarity(syn2)
        if sim is None:
            sim = syn1.path_similarity(syn2)
        if sim is None and syn1.pos() == syn2.pos() == 'n':
            sim = syn1.lin_similarity(syn2, brown_ic)
        return sim if sim is not None else 0.0
    return 0.0

def print_relations(word):
    relations = get_wordnet_relations(word)
    print(f"WordNet relations for '{word}':\n")
    for rel, items in relations.items():
        print(f"{rel.capitalize()}:")
        for item in items:
            print(f"  - {item}")
        print()

def play_word_association_game():
    print("\n=== Word Association Game ===")
    original = input("Enter the original word: ").strip().lower()
    print(f"You entered: {original}")
    print("Try to come up with a word related to it!")
    player_word = input("Your associated word: ").strip().lower()

    # Compute similarity
    similarity = word_similarity(original, player_word)
    points = int(similarity * 100) if similarity else 0

    # Get relations for feedback
    relations = get_wordnet_relations(original)
    related_words = set()
    for rel in ['synonyms', 'antonyms', 'hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
        related_words.update(relations[rel])

    print("\n=== Results ===")
    print(f"Original word: {original}")
    print(f"Your word: {player_word}")
    print(f"Similarity score: {similarity:.2f}")
    print(f"Points awarded: {points}")
    print()
    if player_word in related_words:
        print("🎉 Great! Your word is directly related via WordNet relations.")
    elif similarity > 0.7:
        print("🌟 Excellent! Your word is very close in meaning.")
    elif similarity > 0.4:
        print("👍 Good! Your word is somewhat related.")
    elif similarity > 0:
        print("🙂 Your word is weakly related.")
    else:
        print("❌ No strong relation found. Try again!")
    print("\nWordNet relations for reference:")
    for rel, items in relations.items():
        print(f"{rel.capitalize()}:")
        for item in items:
            print(f"  - {item}")
        print()

class WordNetApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WordNet Explorer & Association Game")
        self.root.geometry("600x600")
        self.create_widgets()

    def create_widgets(self):
        self.root.configure(bg="#e6ffe6")
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TLabel', background="#e6ffe6", foreground="#006633")
        style.configure('TButton', background="#b3ffb3", foreground="#006633")
        style.configure('TEntry', fieldbackground="#f0fff0", foreground="#006633")
        style.configure('TLabelframe', background="#ccffcc", foreground="#006633")
        style.configure('TRadiobutton', background="#e6ffe6", foreground="#006633")

        self.mode_var = tk.StringVar(value="relations")
        mode_frame = ttk.LabelFrame(self.root, text="Choose Mode", style='TLabelframe')
        mode_frame.pack(fill="x", padx=10, pady=10)
        ttk.Radiobutton(mode_frame, text="WordNet Relations", variable=self.mode_var, value="relations", command=self.toggle_assoc_entry, style='TRadiobutton').pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Word Association Game", variable=self.mode_var, value="game", command=self.toggle_assoc_entry, style='TRadiobutton').pack(side="left", padx=5)

        self.input_frame = ttk.LabelFrame(self.root, text="Input", style='TLabelframe')
        self.input_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(self.input_frame, text="Original word:", style='TLabel').grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.word_entry = ttk.Entry(self.input_frame, width=30)
        self.word_entry.grid(row=0, column=1, padx=5, pady=5)
        self.assoc_label = ttk.Label(self.input_frame, text="Associated word (for game):", style='TLabel')
        self.assoc_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.assoc_entry = ttk.Entry(self.input_frame, width=30)
        self.assoc_entry.grid(row=1, column=1, padx=5, pady=5)

        self.run_btn = ttk.Button(self.root, text="Run", command=self.run, style='TButton')
        self.run_btn.pack(pady=10)

        self.output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=70, height=20, bg="#f0fff0", fg="#006633")
        self.output.pack(padx=10, pady=10, fill="both", expand=True)

        self.toggle_assoc_entry()  # Set initial state

    def toggle_assoc_entry(self):
        mode = self.mode_var.get()
        if mode == "relations":
            self.assoc_label.grid_remove()
            self.assoc_entry.grid_remove()
        else:
            self.assoc_label.grid()
            self.assoc_entry.grid()

    def run(self):
        mode = self.mode_var.get()
        word = self.word_entry.get().strip().lower()
        assoc = self.assoc_entry.get().strip().lower()
        self.output.delete(1.0, tk.END)
        if not word:
            messagebox.showerror("Input Error", "Please enter a word.")
            return
        if mode == "relations":
            relations = get_wordnet_relations(word)
            self.output.insert(tk.END, f"WordNet relations for '{word}':\n\n")
            for rel, items in relations.items():
                self.output.insert(tk.END, f"{rel.capitalize()}:\n")
                for item in list(items)[:5]:
                    self.output.insert(tk.END, f"  - {item}\n")
                if len(items) > 5:
                    self.output.insert(tk.END, f"  ...and {len(items)-5} more\n")
                self.output.insert(tk.END, "\n")
        else:
            if not assoc:
                messagebox.showerror("Input Error", "Please enter an associated word for the game.")
                return
            similarity = word_similarity(word, assoc)
            points = int(similarity * 100) if similarity else 0
            relations = get_wordnet_relations(word)
            related_words = set()
            for rel in ['synonyms', 'antonyms', 'hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
                related_words.update(relations[rel])
            self.output.insert(tk.END, f"Similarity score: {similarity:.2f}\n")
            self.output.insert(tk.END, f"Points awarded: {points}\n")
            if assoc in related_words:
                self.output.insert(tk.END, "🎉 Great! Your word is directly related via WordNet relations.\n")
            elif similarity > 0.7:
                self.output.insert(tk.END, "🌟 Excellent! Your word is very close in meaning.\n")
            elif similarity > 0.4:
                self.output.insert(tk.END, "👍 Good! Your word is somewhat related.\n")
            elif similarity > 0:
                self.output.insert(tk.END, "🙂 Your word is weakly related.\n")
            else:
                self.output.insert(tk.END, "❌ No strong relation found. Try again!\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = WordNetApp(root)
    root.mainloop()