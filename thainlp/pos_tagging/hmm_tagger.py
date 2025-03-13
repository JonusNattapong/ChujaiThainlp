"""
Hidden Markov Model for Thai Part-of-Speech Tagging
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

class HMMTagger:
    def __init__(self):
        """Initialize HMM Tagger"""
        self.transition_prob = defaultdict(lambda: defaultdict(float))
        self.emission_prob = defaultdict(lambda: defaultdict(float))
        self.tag_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.tag_word_freq = defaultdict(lambda: defaultdict(int))
        
    def train(self, training_data: List[List[Tuple[str, str]]]):
        """
        Train HMM model on tagged data
        
        Args:
            training_data (List[List[Tuple[str, str]]]): List of sentences, each containing (word, tag) pairs
        """
        # Count frequencies
        for sentence in training_data:
            prev_tag = 'START'
            for word, tag in sentence:
                self.tag_freq[tag] += 1
                self.word_freq[word] += 1
                self.tag_word_freq[tag][word] += 1
                self.transition_prob[prev_tag][tag] += 1
                prev_tag = tag
            self.transition_prob[prev_tag]['END'] += 1
            
        # Calculate probabilities
        for prev_tag in self.transition_prob:
            total = sum(self.transition_prob[prev_tag].values())
            for tag in self.transition_prob[prev_tag]:
                self.transition_prob[prev_tag][tag] /= total
                
        for tag in self.tag_word_freq:
            total = sum(self.tag_word_freq[tag].values())
            for word in self.tag_word_freq[tag]:
                self.emission_prob[tag][word] = self.tag_word_freq[tag][word] / total
                
    def _viterbi(self, sentence: List[str]) -> List[str]:
        """
        Viterbi algorithm for finding most likely tag sequence
        
        Args:
            sentence (List[str]): List of words
            
        Returns:
            List[str]: List of tags
        """
        V = [defaultdict(float)]
        backpointer = [defaultdict(str)]
        
        # Initialize first layer
        for tag in self.tag_freq:
            if tag != 'START':
                V[0][tag] = self.transition_prob['START'][tag] * self.emission_prob[tag].get(sentence[0], 1e-10)
                
        # Forward pass
        for t in range(1, len(sentence)):
            V.append(defaultdict(float))
            backpointer.append(defaultdict(str))
            
            for curr_tag in self.tag_freq:
                if curr_tag == 'START':
                    continue
                    
                max_prob = float('-inf')
                max_prev_tag = None
                
                for prev_tag in self.tag_freq:
                    if prev_tag == 'START':
                        continue
                        
                    prob = V[t-1][prev_tag] * self.transition_prob[prev_tag][curr_tag] * \
                           self.emission_prob[curr_tag].get(sentence[t], 1e-10)
                           
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_tag = prev_tag
                        
                V[t][curr_tag] = max_prob
                backpointer[t][curr_tag] = max_prev_tag
                
        # Backward pass
        tags = []
        curr_tag = max(V[-1].items(), key=lambda x: x[1])[0]
        tags.append(curr_tag)
        
        for t in range(len(sentence)-1, 0, -1):
            curr_tag = backpointer[t][curr_tag]
            tags.append(curr_tag)
            
        return list(reversed(tags))
        
    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a sentence with part-of-speech tags
        
        Args:
            sentence (List[str]): List of words
            
        Returns:
            List[Tuple[str, str]]: List of (word, tag) pairs
        """
        tags = self._viterbi(sentence)
        return list(zip(sentence, tags))

def create_sample_data() -> List[List[Tuple[str, str]]]:
    """
    Create sample training data
    
    Returns:
        List[List[Tuple[str, str]]]: Sample training data
    """
    return [
        [("ผม", "PRON"), ("กิน", "VERB"), ("ข้าว", "NOUN")],
        [("คุณ", "PRON"), ("สวย", "ADJ"), ("มาก", "ADV")],
        [("เขา", "PRON"), ("กำลัง", "AUX"), ("เดิน", "VERB")],
        [("วันนี้", "NOUN"), ("อากาศ", "NOUN"), ("ดี", "ADJ")],
        [("ผม", "PRON"), ("ชอบ", "VERB"), ("กิน", "VERB"), ("ข้าว", "NOUN")],
    ]

def train_and_tag(sentence: List[str]) -> List[Tuple[str, str]]:
    """
    Train HMM model and tag a sentence
    
    Args:
        sentence (List[str]): Sentence to tag
        
    Returns:
        List[Tuple[str, str]]: Tagged sentence
    """
    tagger = HMMTagger()
    training_data = create_sample_data()
    tagger.train(training_data)
    return tagger.tag(sentence) 