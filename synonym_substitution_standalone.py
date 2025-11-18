"""
Standalone Synonym Substitution Transformation

This module provides a standalone implementation of synonym substitution for text augmentation.
It substitutes words with their synonyms using spaCy for POS tagging and WordNet via NLTK for synonyms.

Original implementation from: nlaugmenter/transformations/synonym_substitution
Author: Zijian Wang (zijwang@hotmail.com)

Usage:
    from synonym_substitution_standalone import SynonymSubstitution
    
    transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
    results = transformer.generate("Andrew finally returned the French book to Chris.")
    print(results)
"""

import re
import nltk
import numpy as np
import spacy
from nltk.corpus import wordnet
from typing import List


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    ref: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py#L28
    """
    text = " ".join(words)
    step1 = (
        text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
    )
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = (
        step4.replace(" '", "'")
        .replace(" n't", "n't")
        .replace("can not", "cannot")
    )
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def synonym_substitution(
    text, spacy_pipeline, seed=42, prob=0.5, max_outputs=1
):
    """
    Substitute words with synonyms using spaCy (for POS) and WordNet via NLTK (for synonyms).
    
    Args:
        text (str): Input text to transform
        spacy_pipeline: Loaded spaCy language model
        seed (int): Random seed for reproducibility
        prob (float): Probability of substituting a word (0.0 to 1.0)
        max_outputs (int): Maximum number of output variations to generate
    
    Returns:
        List[str]: List of transformed text variations
    """
    np.random.seed(seed)
    upos_wn_dict = {
        "VERB": "v",
        "NOUN": "n",
        "ADV": "r",
        "ADJ": "s",
    }

    doc = spacy_pipeline(text)
    results = []
    for _ in range(max_outputs):
        result = []
        for token in doc:
            word = token.text
            wn_pos = upos_wn_dict.get(token.pos_)
            if wn_pos is None:
                result.append(word)
            else:
                syns = wordnet.synsets(word, pos=wn_pos)
                syns = [syn.name().split(".")[0] for syn in syns]
                syns = [syn for syn in syns if syn.lower() != word.lower()]
                if len(syns) > 0 and np.random.random() < prob:
                    result.append(np.random.choice(syns).replace("_", " "))
                else:
                    result.append(word)

        # detokenize sentences
        result = untokenize(result)
        if result not in results:
            # make sure there is no dup in results
            results.append(result)
    return results


class SynonymSubstitution:
    """
    Substitute words with synonyms using spaCy (for POS) and WordNet via NLTK (for synonyms).
    
    This transformation could augment the semantic representation of the sentence as well as 
    test model robustness by substituting words with their synonyms.
    
    Attributes:
        nlp: Loaded spaCy language model
        seed (int): Random seed for reproducibility
        prob (float): Probability of substituting a word (0.0 to 1.0)
        max_outputs (int): Maximum number of output variations to generate
    
    Example:
        >>> transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
        >>> results = transformer.generate("Andrew finally returned the French book to Chris.")
        >>> print(results)
    """
    
    def __init__(self, seed=42, prob=0.5, max_outputs=1, spacy_model="en_core_web_sm"):
        """
        Initialize the SynonymSubstitution transformer.
        
        Args:
            seed (int): Random seed for reproducibility (default: 42)
            prob (float): Probability of substituting a word (default: 0.5)
            max_outputs (int): Maximum number of output variations (default: 1)
            spacy_model (str): spaCy model to use (default: "en_core_web_sm")
        """
        self.seed = seed
        self.prob = prob
        self.max_outputs = max_outputs
        
        # Download WordNet if not already downloaded
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Please install it using: python -m spacy download {spacy_model}"
            )

    def generate(self, sentence: str) -> List[str]:
        """
        Generate synonym-substituted variations of the input sentence.
        
        Args:
            sentence (str): Input sentence to transform
        
        Returns:
            List[str]: List of transformed sentence variations
        """
        perturbed = synonym_substitution(
            text=sentence,
            spacy_pipeline=self.nlp,
            seed=self.seed,
            prob=self.prob,
            max_outputs=self.max_outputs,
        )
        return perturbed


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    print("Example 1: Basic usage")
    transformer = SynonymSubstitution(seed=42, prob=0.5, max_outputs=1)
    sentence = "Andrew finally returned the French book to Chris that I bought last week."
    results = transformer.generate(sentence)
    print(f"Original: {sentence}")
    print(f"Transformed: {results[0] if results else 'No transformation'}")
    print()
    
    # Example 2: Multiple outputs
    print("Example 2: Multiple outputs")
    transformer2 = SynonymSubstitution(seed=42, prob=0.5, max_outputs=3)
    sentence2 = "The quick brown fox jumps over the lazy dog."
    results2 = transformer2.generate(sentence2)
    print(f"Original: {sentence2}")
    for i, result in enumerate(results2, 1):
        print(f"Variant {i}: {result}")
    print()
    
    # Example 3: Higher probability
    print("Example 3: Higher substitution probability")
    transformer3 = SynonymSubstitution(seed=42, prob=0.8, max_outputs=1)
    sentence3 = "Sentences with gapping, such as Paul likes coffee and Mary tea."
    results3 = transformer3.generate(sentence3)
    print(f"Original: {sentence3}")
    print(f"Transformed: {results3[0] if results3 else 'No transformation'}")

