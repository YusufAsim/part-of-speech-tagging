import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict, Counter

nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

class HMMTagger:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.emissions = defaultdict(Counter)
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()

    def train(self, tagged_sents):
        # TO DO: Iterate through the tagged sentences. Implement training logic here
        pass

    def get_emission_prob(self, word, tag):
        # TO DO: Implement emission probability calculation.
        return 0.0

    def get_transition_prob(self, prev_tag, curr_tag):
        # TO DO: Implement transition probability calculation.
        return 0.0

    def viterbi(self, sentence):
        # TO DO: Implement Viterbi algorithm to find the most likely tag sequence
        return []

def main():

    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    print(f"Total sentences in dataset: {len(tagged_sents)}")
    print(f"\nFirst example sentence:\n{tagged_sents[0]}")
    
    universal_tags = sorted(set(tag for sent in tagged_sents for word, tag in sent))
    print(f"\nUniversal Tags: {universal_tags}")

    train_sents, test_sents = train_test_split(tagged_sents, test_size=0.2, random_state=42)
    
    print(f"\nWorking with Subset -> Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")
    
    """
    tagger = HMMTagger()
    tagger.train(train_sents)

    transitions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.transitions.items())[:5]}
    emissions_summary = {k: dict(sorted(v.items())[:5]) for k, v in sorted(tagger.emissions.items())[:5]}

    print(f"\nTagger Transitions (first 5 sorted): {transitions_summary}")
    print(f"\nTagger Emissions (first 5 sorted): {emissions_summary}")
    print(f"\nTagger Tags (sorted): {sorted(list(tagger.tags))}")
    print(f"\nVocabulary size: {len(tagger.vocab)}")
    print(f"\nTagger Vocab (first 5 sorted): {sorted(list(tagger.vocab))[:5]}")
    print(f"\nTagger Tag Counts (sorted items): {dict(sorted(tagger.tag_counts.items()))}")

    print("\n=== VITERBI DECODING & EVALUATION ===")
    y_true = []
    y_pred = []
    last_5_samples = []

    print("Starting decoding for test sentences...")
    for i, sent in enumerate(test_sents):
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        
        pred_tags = tagger.viterbi(words)
        
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)
        
        if i >= len(test_sents) - 5:
            last_5_samples.append((words, true_tags, pred_tags))
        
        if (i + 1) % 2500 == 0:
            print(f"Processed {i+1}/{len(test_sents)} sentences...")

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("=== SAMPLE PREDICTIONS (LAST 5 SENTENCES) ===")
    for words, true, pred in last_5_samples:
        print("-" * 50)
        print(f"SENTENCE  : {' '.join(words)}")
        print(f"ORIGINAL  : {true}")
        print(f"PREDICTED : {pred}")
    """

if __name__ == "__main__":
    main()
