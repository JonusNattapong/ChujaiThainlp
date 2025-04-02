"""
Hidden Markov Model based POS tagger for Thai
"""
from typing import List, Tuple, Dict
import json
from collections import defaultdict

class HMMTagger:
    def __init__(self):
        # Basic tag set
        self.tags = {
            'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 
            'CONJ', 'PART', 'NUM', 'PUNCT'
        }
        
        # Initial probabilities
        self.initial_prob = defaultdict(float)
        self.initial_prob.update({
            'NOUN': 0.3,
            'VERB': 0.2,
            'ADJ': 0.1,
            'PRON': 0.1,
            'DET': 0.1
        })
        
        # Transition probabilities
        self.transitions = defaultdict(lambda: defaultdict(float))
        self._init_transitions()
        
        # Emission probabilities (word -> tag probabilities)
        self.emissions = defaultdict(lambda: defaultdict(float))
        self._init_emissions()
        
    def _init_transitions(self):
        """Initialize transition probabilities"""
        basic_transitions = {
            'NOUN': {'VERB': 0.4, 'ADP': 0.2, 'CONJ': 0.1},
            'VERB': {'NOUN': 0.3, 'ADV': 0.2, 'PRON': 0.1},
            'ADJ': {'NOUN': 0.5, 'CONJ': 0.1},
            'ADV': {'VERB': 0.4, 'ADJ': 0.2},
            'PRON': {'VERB': 0.5, 'ADP': 0.1},
            'DET': {'NOUN': 0.8},
            'ADP': {'NOUN': 0.6, 'PRON': 0.2},
            'CONJ': {'NOUN': 0.3, 'VERB': 0.3, 'ADJ': 0.2},
            'PART': {'VERB': 0.3, 'ADJ': 0.2},
            'NUM': {'NOUN': 0.7},
            'PUNCT': {'NOUN': 0.3, 'VERB': 0.2}
        }
        
        for tag1, transitions in basic_transitions.items():
            for tag2, prob in transitions.items():
                self.transitions[tag1][tag2] = prob
                
    def _init_emissions(self):
        """Initialize emission probabilities with common Thai words"""
        common_words = {
            'NOUN': ['บ้าน', 'รถ', 'คน', 'หมา', 'แมว'],
            'VERB': ['กิน', 'วิ่ง', 'นอน', 'เดิน', 'พูด'],
            'ADJ': ['ดี', 'สวย', 'เร็ว', 'ช้า', 'ใหญ่'],
            'ADV': ['เร็วๆ', 'ช้าๆ', 'ดีๆ', 'มาก', 'น้อย'],
            'PRON': ['ฉัน', 'เขา', 'มัน', 'เธอ', 'พวกเรา'],
            'DET': ['นี้', 'นั้น', 'โน้น', 'นู้น'],
            'ADP': ['ใน', 'บน', 'ใต้', 'นอก', 'ระหว่าง'],
            'CONJ': ['และ', 'หรือ', 'แต่', 'เพราะ'],
            'PART': ['ค่ะ', 'ครับ', 'จ้ะ', 'จ้า'],
            'NUM': ['หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า'],
            'PUNCT': ['.', ',', '!', '?', ';']
        }
        
        for tag, words in common_words.items():
            prob = 1.0 / len(words)
            for word in words:
                self.emissions[word][tag] = prob
                
    def tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Tag a sequence of tokens using Viterbi algorithm"""
        if not tokens:
            return []
            
        # Initialize Viterbi variables
        V = [{}]
        path = {}
        
        # Initialize base cases
        for tag in self.tags:
            V[0][tag] = self.initial_prob[tag] * self.emissions[tokens[0]][tag]
            path[tag] = [tag]
            
        # Run Viterbi for subsequent tokens
        for t in range(1, len(tokens)):
            V.append({})
            newpath = {}
            
            for tag in self.tags:
                # Find the best previous tag
                prob_tag = []
                for prev_tag in self.tags:
                    prob = V[t-1][prev_tag] * \
                          self.transitions[prev_tag][tag] * \
                          self.emissions[tokens[t]][tag]
                    prob_tag.append((prob, prev_tag))
                    
                best_prob, best_tag = max(prob_tag)
                V[t][tag] = best_prob
                newpath[tag] = path[best_tag] + [tag]
                
            path = newpath
            
        # Find the best path
        prob_tag = [(V[len(tokens)-1][tag], tag) for tag in self.tags]
        _, best_tag = max(prob_tag)
        
        return list(zip(tokens, path[best_tag]))
        
    def save(self, filepath: str):
        """Save model parameters to file"""
        data = {
            'initial_prob': dict(self.initial_prob),
            'transitions': {k: dict(v) for k, v in self.transitions.items()},
            'emissions': {k: dict(v) for k, v in self.emissions.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath: str):
        """Load model parameters from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.initial_prob = defaultdict(float, data['initial_prob'])
        
        self.transitions = defaultdict(lambda: defaultdict(float))
        for tag1, trans in data['transitions'].items():
            for tag2, prob in trans.items():
                self.transitions[tag1][tag2] = prob
                
        self.emissions = defaultdict(lambda: defaultdict(float))
        for word, emis in data['emissions'].items():
            for tag, prob in emis.items():
                self.emissions[word][tag] = prob