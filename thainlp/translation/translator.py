"""
Thai Translation Module using PyThaiNLP with enhanced dictionary and context handling
Version: 0.3.0
"""

from typing import Dict, List, Union, Optional, Tuple, Any
import re
import warnings
from collections import defaultdict

try:
    import pythainlp
    from pythainlp.translate import Translator
    from pythainlp.tokenize import word_tokenize, Tokenizer, syllable_tokenize
    from pythainlp.util import normalize, thai_strftime, num_to_thaiword
    from pythainlp.corpus import thai_words, thai_syllables, thai_stopwords, conceptual_word_list
    from pythainlp.tag import pos_tag
    from pythainlp.tokenize import sent_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified translation.")

class ThaiTranslator:
    def __init__(self, engine: str = "default"):
        """
        Initialize ThaiTranslator
        
        Args:
            engine (str): Translation engine to use
                - 'default': Use PyThaiNLP's default translator
                - 'mt': Use Machine Translation
                - 'local': Use local dictionary (fallback)
        """
        self.engine = engine
        
        if PYTHAINLP_AVAILABLE and engine != "local":
            try:
                self.translator = Translator(engine=engine)
                # Initialize PyThaiNLP's word tokenizer with custom dictionary
                self.word_tokenizer = Tokenizer(custom_dict=thai_words())
            except Exception as e:
                warnings.warn(f"Failed to initialize PyThaiNLP translator: {e}. Falling back to local dictionary.")
                self.engine = "local"
        else:
            self.engine = "local"
            
        if self.engine == "local":
            self._init_local_dict()
            
        # Initialize context tracking
        self.context = {
            'formality': 'neutral',  # neutral, formal, informal
            'gender': None,  # male, female, neutral
            'plurality': 'singular',  # singular, plural
            'tense': 'present',  # past, present, future
            'relationship': 'neutral',  # superior, peer, inferior
            'social_context': 'general',  # formal_meeting, casual_meeting, family, service
            'emotion': 'neutral',  # polite, friendly, urgent, apologetic
            'dialect': 'central',  # central, north, northeast, south
            'age_group': 'adult',  # child, teen, adult, elder
            'domain': 'general',  # academic, business, medical, legal, technical
            'register': 'neutral',  # formal_written, formal_spoken, informal_written, informal_spoken
            'honorific_level': 0,  # 0-5, where 0 is informal and 5 is most formal
        }
        
        # Initialize additional components
        if PYTHAINLP_AVAILABLE:
            self._init_advanced_components()
    
    def _init_local_dict(self):
        """Initialize enhanced local translation dictionary"""
        # Load base dictionaries
        self._load_base_dictionaries()
        
        # Initialize grammar components
        self._init_grammar_components()
        
        # Initialize context-specific translations
        self._init_context_translations()
    
    def _load_base_dictionaries(self):
        """Load comprehensive base dictionaries"""
        # Load existing dictionary (previous implementation)
        self.th_to_en = {
            # Common words
            'สวัสดี': 'hello',
            'ขอบคุณ': 'thank you',
            'ใช่': 'yes',
            'ไม่': 'no',
            'กิน': 'eat',
            'นอน': 'sleep',
            'ไป': 'go',
            'มา': 'come',
            'ดี': 'good',
            'ไม่ดี': 'bad',
            'ใหญ่': 'big',
            'เล็ก': 'small',
            'เร็ว': 'fast',
            'ช้า': 'slow',
            'ร้อน': 'hot',
            'เย็น': 'cold',
            'น้ำ': 'water',
            'อาหาร': 'food',
            'บ้าน': 'house',
            'รถ': 'car',
            'คน': 'person',
            'เงิน': 'money',
            'เวลา': 'time',
            'วัน': 'day',
            'คืน': 'night',
            'ปี': 'year',
            'เดือน': 'month',
            
            # Pronouns
            'ฉัน': 'I',
            'ผม': 'I (male)',
            'ดิฉัน': 'I (female)',
            'คุณ': 'you',
            'เขา': 'he/she',
            'เรา': 'we',
            'พวกเขา': 'they',
            
            # Numbers
            'หนึ่ง': 'one',
            'สอง': 'two',
            'สาม': 'three',
            'สี่': 'four',
            'ห้า': 'five',
            'หก': 'six',
            'เจ็ด': 'seven',
            'แปด': 'eight',
            'เก้า': 'nine',
            'สิบ': 'ten',
            'ร้อย': 'hundred',
            'พัน': 'thousand',
            'หมื่น': 'ten thousand',
            'แสน': 'hundred thousand',
            'ล้าน': 'million',
            
            # Common phrases
            'สบายดีไหม': 'how are you',
            'ไม่เป็นไร': "it's okay",
            'ยินดีที่ได้รู้จัก': 'nice to meet you',
            'เข้าใจไหม': 'do you understand',
            'พูดช้าๆ': 'speak slowly',
            'ราคาเท่าไร': 'how much',
            'ไปที่ไหน': 'where are you going',
            'มาจากไหน': 'where are you from',
            'ช่วยด้วย': 'help',
            'ขอโทษ': 'sorry',
        }
        
        # English to Thai dictionary (reverse of th_to_en)
        self.en_to_th = {v: k for k, v in self.th_to_en.items()}
        
        # Add some special cases for English to Thai
        self.en_to_th.update({
            'hello': 'สวัสดี',
            'thank you': 'ขอบคุณ',
            'yes': 'ใช่',
            'no': 'ไม่',
            'i': 'ฉัน',
            'you': 'คุณ',
            'he': 'เขา',
            'she': 'เขา',
            'we': 'เรา',
            'they': 'พวกเขา',
        })
        
        # Thai grammar patterns (simplified)
        self.th_grammar = {
            'subject_verb_object': '{subject}{verb}{object}',
            'subject_verb': '{subject}{verb}',
            'verb_object': '{verb}{object}',
            'adjective_noun': '{noun}{adjective}',
        }
        
        # English grammar patterns (simplified)
        self.en_grammar = {
            'subject_verb_object': '{subject} {verb} {object}',
            'subject_verb': '{subject} {verb}',
            'verb_object': '{verb} {object}',
            'adjective_noun': '{adjective} {noun}',
        }
        
        # Thai particles
        self.th_particles = ['ครับ', 'ค่ะ', 'นะ', 'สิ', 'เถอะ', 'เลย', 'แล้ว', 'อยู่', 'ด้วย']
        
        # Add classifiers
        self.classifiers = {
            'คน': ['person', 'people'],
            'ตัว': ['animal', 'body', 'piece'],
            'อัน': ['thing', 'piece'],
            'เล่ม': ['book', 'notebook'],
            'คัน': ['vehicle'],
            'ลูก': ['ball', 'fruit', 'child'],
            'ชิ้น': ['piece', 'item'],
            'แก้ว': ['glass', 'cup'],
            'จาน': ['plate', 'dish'],
            'ใบ': ['leaf', 'paper'],
            'เม็ด': ['pill', 'grain'],
            'ก้อน': ['lump', 'piece'],
            'แผ่น': ['sheet', 'slice'],
            'เส้น': ['line', 'noodle', 'string'],
        }
        
        # Add tone markers and their effects
        self.tone_markers = {
            '่': 'low tone',
            '้': 'falling tone',
            '๊': 'high tone',
            '๋': 'rising tone',
            '': 'mid tone',
        }
        
        # Add particles and their contextual meanings
        self.particles = {
            'ครับ': {
                'meaning': 'polite particle (male)',
                'formality': 'formal',
                'gender': 'male',
            },
            'ค่ะ': {
                'meaning': 'polite particle (female)',
                'formality': 'formal',
                'gender': 'female',
            },
            'ฮะ': {
                'meaning': 'casual particle',
                'formality': 'informal',
            },
            'จ้ะ': {
                'meaning': 'endearing particle',
                'formality': 'informal',
            },
            'นะ': {
                'meaning': 'softening particle',
                'formality': 'neutral',
            },
            'สิ': {
                'meaning': 'emphatic particle',
                'formality': 'informal',
            },
            'เถอะ': {
                'meaning': 'persuasive particle',
                'formality': 'informal',
            },
            'ล่ะ': {
                'meaning': 'questioning particle',
                'formality': 'neutral',
            },
            'เลย': {
                'meaning': 'intensifying particle',
                'formality': 'informal',
            },
        }
        
        # Add time-related words and expressions
        self.time_expressions = {
            'เมื่อวาน': 'yesterday',
            'วันนี้': 'today',
            'พรุ่งนี้': 'tomorrow',
            'เมื่อคืน': 'last night',
            'คืนนี้': 'tonight',
            'เช้านี้': 'this morning',
            'บ่ายนี้': 'this afternoon',
            'เย็นนี้': 'this evening',
            'สัปดาห์ที่แล้ว': 'last week',
            'สัปดาห์หน้า': 'next week',
            'เดือนที่แล้ว': 'last month',
            'เดือนหน้า': 'next month',
            'ปีที่แล้ว': 'last year',
            'ปีหน้า': 'next year',
        }
        
        # Add common verb forms
        self.verb_forms = {
            'กำลัง': {
                'meaning': 'progressive aspect',
                'tense': 'present',
            },
            'จะ': {
                'meaning': 'future marker',
                'tense': 'future',
            },
            'ได้': {
                'meaning': 'past marker/ability',
                'tense': 'past',
            },
            'เคย': {
                'meaning': 'experiential marker',
                'tense': 'past',
            },
        }
    
    def _init_advanced_components(self):
        """Initialize advanced components when PyThaiNLP is available"""
        # Load conceptual word groups
        self.conceptual_words = conceptual_word_list()
        
        # Initialize sentence tokenizer
        self.sent_tokenizer = sent_tokenize
        
        # Initialize syllable tokenizer
        self.syllable_tokenizer = syllable_tokenize
        
        # Add number converter
        self.number_converter = num_to_thaiword
    
    def _init_grammar_components(self):
        """Initialize enhanced grammar components"""
        # Thai sentence patterns
        self.th_patterns = {
            'basic': {
                'statement': '{subject}{verb}{object}',
                'question': '{question_word}{subject}{verb}{object}หรือ',
                'negative': '{subject}ไม่{verb}{object}',
            },
            'complex': {
                'conditional': 'ถ้า{condition}ก็{result}',
                'causative': 'เพราะ{cause}จึง{effect}',
                'temporal': 'เมื่อ{time}{subject}{verb}{object}',
            },
        }
        
        # Word order rules
        self.word_order = {
            'basic': ['subject', 'verb', 'object'],
            'modifiers': ['noun', 'adjective'],
            'time': ['time_expression', 'subject', 'verb', 'object'],
        }
        
        # Aspect markers and their positions
        self.aspect_markers = {
            'กำลัง': {'position': 'pre-verb', 'meaning': 'progressive'},
            'แล้ว': {'position': 'post-verb', 'meaning': 'perfective'},
            'อยู่': {'position': 'post-verb', 'meaning': 'continuous'},
        }
        
        # Add complex sentence structures
        self.th_patterns.update({
            'passive': {
                'pattern': '{object}ถูก{subject}{verb}',
                'conditions': {'voice': 'passive'}
            },
            'causative': {
                'pattern': '{causer}ทำให้{causee}{verb}',
                'conditions': {'causation': True}
            },
            'serial_verb': {
                'pattern': '{verb1}ไป{verb2}',
                'conditions': {'serial_verbs': True}
            },
            'topic_comment': {
                'pattern': '{topic}น่ะ {comment}',
                'conditions': {'topic_marking': True}
            }
        })
        
        # Add honorific prefixes
        self.honorifics = {
            'superior': {
                'person': 'ท่าน',
                'action': 'กรุณา',
                'possession': 'ของท่าน'
            },
            'peer': {
                'person': 'คุณ',
                'action': 'ช่วย',
                'possession': 'ของคุณ'
            },
            'inferior': {
                'person': '',
                'action': '',
                'possession': 'ของ'
            }
        }
        
        # Add social context patterns
        self.social_patterns = {
            'formal_meeting': {
                'greeting': 'สวัสดีครับ/ค่ะ ท่าน{title}',
                'request': 'ขออนุญาต{action}',
                'farewell': 'ขอบคุณครับ/ค่ะ'
            },
            'casual_meeting': {
                'greeting': 'สวัสดี{particle}',
                'request': '{action}หน่อย',
                'farewell': 'ลาก่อน{particle}'
            },
            'family': {
                'greeting': '{term_of_address}',
                'request': '{action}หน่อย',
                'farewell': 'บาย{particle}'
            },
            'service': {
                'greeting': 'สวัสดี{particle} ยินดีให้บริการ',
                'request': 'ขอ{action}',
                'farewell': 'ขอบคุณที่ใช้บริการ{particle}'
            }
        }
        
        # Add aspect and mood markers with positions
        self.aspect_mood_markers = {
            'progressive': {
                'marker': 'กำลัง',
                'position': 'pre-verb',
                'conditions': {'tense': 'present', 'aspect': 'continuous'}
            },
            'perfective': {
                'marker': 'แล้ว',
                'position': 'post-verb',
                'conditions': {'tense': 'past', 'aspect': 'completed'}
            },
            'future': {
                'marker': 'จะ',
                'position': 'pre-verb',
                'conditions': {'tense': 'future'}
            },
            'experiential': {
                'marker': 'เคย',
                'position': 'pre-verb',
                'conditions': {'aspect': 'experiential'}
            },
            'habitual': {
                'marker': 'ชอบ',
                'position': 'pre-verb',
                'conditions': {'aspect': 'habitual'}
            }
        }
        
        # Add classifier usage patterns
        self.classifier_patterns = {
            'counting': '{number}{classifier}',
            'demonstrative': '{noun}{classifier}{demonstrative}',
            'quantity': '{quantity}{classifier}',
            'specific': '{noun}{classifier}{specific_marker}'
        }
        
        # Add particle combinations
        self.particle_combinations = {
            'polite_request': ['ครับ/ค่ะ', 'นะ'],
            'gentle_suggestion': ['นะ', 'คะ/ครับ'],
            'strong_suggestion': ['สิ', 'คะ/ครับ'],
            'friendly_question': ['หรือ', 'เปล่า'],
            'polite_question': ['หรือ', 'ไหม', 'คะ/ครับ']
        }
        
        # Add dialect-specific patterns
        self.dialect_patterns = {
            'north': {
                'particles': ['จ้าว', 'เจ้า', 'กอ'],
                'pronouns': {'I': 'อั๋น', 'you': 'เจ้า'},
                'verbs': {'eat': 'กิ๋น', 'go': 'ไป๋'},
            },
            'northeast': {
                'particles': ['เด้อ', 'อีหลี', 'สิ'],
                'pronouns': {'I': 'ข้อย', 'you': 'เจ้า'},
                'verbs': {'eat': 'กิน', 'go': 'ไป'},
            },
            'south': {
                'particles': ['ว่ะ', 'หวา', 'โว้ย'],
                'pronouns': {'I': 'ฉาน', 'you': 'มึง'},
                'verbs': {'eat': 'แดก', 'go': 'ไป'},
            },
        }
        
        # Add domain-specific patterns
        self.domain_patterns = {
            'academic': {
                'prefix': ['ดังนั้น', 'กล่าวคือ', 'โดยสรุป'],
                'verbs': ['วิเคราะห์', 'สังเคราะห์', 'อภิปราย'],
                'style': 'formal_written',
            },
            'business': {
                'prefix': ['เรียน', 'ขอแจ้งให้ทราบว่า', 'ขอความกรุณา'],
                'verbs': ['ดำเนินการ', 'พิจารณา', 'อนุมัติ'],
                'style': 'formal_written',
            },
            'medical': {
                'prefix': ['อาการ', 'การรักษา', 'การวินิจฉัย'],
                'verbs': ['ตรวจ', 'รักษา', 'ผ่าตัด'],
                'style': 'formal_spoken',
            },
            'legal': {
                'prefix': ['ด้วยเหตุนี้', 'อาศัยอำนาจตาม', 'พิจารณาแล้ว'],
                'verbs': ['พิพากษา', 'ตัดสิน', 'บังคับ'],
                'style': 'formal_written',
            },
        }
        
        # Add age-specific patterns
        self.age_patterns = {
            'child': {
                'vocabulary_level': 'simple',
                'sentence_length': 'short',
                'particles': ['จ้ะ', 'ค่ะ', 'ครับ'],
            },
            'teen': {
                'vocabulary_level': 'moderate',
                'sentence_length': 'medium',
                'particles': ['อ่ะ', 'น่ะ', 'อ่ะ'],
            },
            'adult': {
                'vocabulary_level': 'advanced',
                'sentence_length': 'long',
                'particles': ['ครับ', 'ค่ะ', 'นะ'],
            },
            'elder': {
                'vocabulary_level': 'traditional',
                'sentence_length': 'medium',
                'particles': ['จ้ะ', 'เจ้าค่ะ', 'ขอรับ'],
            },
        }
        
        # Add register-specific patterns
        self.register_patterns = {
            'formal_written': {
                'style': 'academic',
                'particles': [''],
                'conjunctions': ['และ', 'หรือ', 'แต่'],
            },
            'formal_spoken': {
                'style': 'polite',
                'particles': ['ครับ', 'ค่ะ'],
                'conjunctions': ['และ', 'หรือ'],
            },
            'informal_written': {
                'style': 'casual',
                'particles': ['นะ', 'ค่ะ', 'ครับ'],
                'conjunctions': ['แล้วก็', 'หรือว่า'],
            },
            'informal_spoken': {
                'style': 'colloquial',
                'particles': ['อ่ะ', 'น่ะ', 'สิ'],
                'conjunctions': ['ก็', 'แล้วก็'],
            },
        }
        
        # Add honorific levels
        self.honorific_levels = {
            0: {  # Informal
                'pronouns': {'I': 'กู', 'you': 'มึง'},
                'particles': ['ว่ะ', 'โว้ย'],
                'verbs': {'eat': 'แดก', 'sleep': 'นอน'},
            },
            1: {  # Casual
                'pronouns': {'I': 'ฉัน', 'you': 'เธอ'},
                'particles': ['จ้ะ', 'จ้า'],
                'verbs': {'eat': 'กิน', 'sleep': 'นอน'},
            },
            2: {  # Polite
                'pronouns': {'I': 'ผม/ดิฉัน', 'you': 'คุณ'},
                'particles': ['ครับ', 'ค่ะ'],
                'verbs': {'eat': 'ทาน', 'sleep': 'นอน'},
            },
            3: {  # Respectful
                'pronouns': {'I': 'ผม/ดิฉัน', 'you': 'ท่าน'},
                'particles': ['ครับ/ค่ะ'],
                'verbs': {'eat': 'รับประทาน', 'sleep': 'นอนหลับ'},
            },
            4: {  # Formal
                'pronouns': {'I': 'กระผม/ดิฉัน', 'you': 'ท่าน'},
                'particles': ['ขอรับ', 'เจ้าค่ะ'],
                'verbs': {'eat': 'รับประทาน', 'sleep': 'บรรทม'},
            },
            5: {  # Royal
                'pronouns': {'I': 'ข้าพระพุทธเจ้า', 'you': 'ใต้ฝ่าละอองธุลีพระบาท'},
                'particles': ['พระพุทธเจ้าข้า', 'เกล้ากระหม่อม'],
                'verbs': {'eat': 'เสวย', 'sleep': 'บรรทม'},
            },
        }
    
    def _init_context_translations(self):
        """Initialize context-specific translations"""
        # Formality levels
        self.formality_levels = {
            'formal': {
                'I': {'male': 'ผม', 'female': 'ดิฉัน'},
                'you': 'ท่าน',
                'eat': 'รับประทาน',
                'sleep': 'นอนหลับ',
            },
            'neutral': {
                'I': 'ฉัน',
                'you': 'คุณ',
                'eat': 'กิน',
                'sleep': 'นอน',
            },
            'informal': {
                'I': 'กู',
                'you': 'มึง',
                'eat': 'แดก',
                'sleep': 'นอน',
            },
        }
        
        # Context-dependent translations
        self.context_translations = defaultdict(dict)
        for formality in ['formal', 'neutral', 'informal']:
            for gender in ['male', 'female', 'neutral']:
                key = f"{formality}_{gender}"
                self.context_translations[key] = {
                    **self.formality_levels[formality],
                    'particle': self.particles.get(
                        'ครับ' if gender == 'male' else 'ค่ะ' if gender == 'female' else 'ครับ/ค่ะ'
                    ),
                }
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en", context: Optional[Dict] = None) -> Dict[str, Union[str, float]]:
        """
        Enhanced translate function with context awareness
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language ('th', 'en', or 'auto')
            target_lang (str): Target language ('th' or 'en')
            context (Optional[Dict]): Translation context
                - formality: 'formal', 'neutral', 'informal'
                - gender: 'male', 'female', 'neutral'
                - plurality: 'singular', 'plural'
                - tense: 'past', 'present', 'future'
            
        Returns:
            Dict[str, Union[str, float]]: Translation result with confidence score
        """
        if context:
            self.context.update(context)
            
        if not text:
            return {'translation': '', 'score': 0.0}
            
        # Auto-detect language if needed
        if source_lang == "auto":
            source_lang = self.detect_language(text)
            
        if source_lang not in ['th', 'en'] or target_lang not in ['th', 'en']:
            return {
                'translation': text,
                'score': 0.0,
                'error': 'Unsupported language pair. Only th-en and en-th are supported.'
            }
            
        if source_lang == target_lang:
            return {'translation': text, 'score': 1.0}
            
        # Use PyThaiNLP translator if available
        if PYTHAINLP_AVAILABLE and self.engine != "local":
            try:
                translation = self.translator.translate(text, source_lang=source_lang, target_lang=target_lang)
                # Apply context-aware post-processing
                translation = self._post_process_translation(translation, target_lang)
                return {'translation': translation, 'score': 0.8}
            except Exception as e:
                warnings.warn(f"PyThaiNLP translation failed: {e}. Falling back to local dictionary.")
                
        # Fallback to enhanced local dictionary translation
        return self._translate_local_enhanced(text, source_lang, target_lang)
    
    def _translate_local_enhanced(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Union[str, float]]:
        """Enhanced local dictionary-based translation with context awareness"""
        if source_lang == 'th':
            # Use PyThaiNLP's word tokenizer if available
            if PYTHAINLP_AVAILABLE:
                tokens = self.word_tokenizer.word_tokenize(text)
                pos_tags = pos_tag(tokens) if PYTHAINLP_AVAILABLE else None
            else:
                tokens = self._tokenize_thai(text)
                pos_tags = None
                
            translated = []
            for i, token in enumerate(tokens):
                # Get POS tag if available
                pos = pos_tags[i][1] if pos_tags else None
                
                # Handle particles
                if token in self.particles:
                    if target_lang == 'en':
                        # Skip particles in English translation
                        continue
                    particle_info = self.particles[token]
                    if particle_info['formality'] == self.context['formality']:
                        translated.append(token)
                    continue
                
                # Handle classifiers
                if token in self.classifiers:
                    classifier_translations = self.classifiers[token]
                    # Choose appropriate classifier based on context
                    translated.append(classifier_translations[0])
                    continue
                
                # Handle time expressions
                if token in self.time_expressions:
                    translated.append(self.time_expressions[token])
                    continue
                
                # Handle verb forms
                if token in self.verb_forms:
                    verb_info = self.verb_forms[token]
                    if verb_info['tense'] == self.context['tense']:
                        # Apply appropriate verb form
                        translated.append(verb_info['meaning'])
                    continue
                
                # Default translation
                translated_token = self.th_to_en.get(token.lower(), token)
                translated.append(translated_token)
            
            # Post-process translation
            result = ' '.join(translated)
            result = self._post_process_translation(result, target_lang)
            
            # Apply context-specific rules
            result = self._apply_context_rules(result, target_lang)
            
            return {'translation': result, 'score': 0.6}
        else:
            # English to Thai translation
            tokens = text.split()
            translated = []
            
            for token in tokens:
                # Get context-specific translation
                context_key = f"{self.context['formality']}_{self.context['gender']}"
                if token.lower() in self.context_translations[context_key]:
                    translated.append(self.context_translations[context_key][token.lower()])
                    continue
                
                # Handle time expressions
                reversed_time = {v: k for k, v in self.time_expressions.items()}
                if token.lower() in reversed_time:
                    translated.append(reversed_time[token.lower()])
                    continue
                
                # Default translation
                translated_token = self.en_to_th.get(token.lower(), token)
                translated.append(translated_token)
            
            # Add appropriate particle based on context
            if self.context['formality'] != 'informal':
                particle = 'ครับ' if self.context['gender'] == 'male' else 'ค่ะ' if self.context['gender'] == 'female' else ''
                if particle:
                    translated.append(particle)
            
            result = ''.join(translated)
            
            # Apply social context patterns
            if self.context['social_context'] in self.social_patterns:
                pattern = self.social_patterns[self.context['social_context']]
                if text.lower().startswith(('hello', 'hi')):
                    result = pattern['greeting'].format(
                        particle=self._get_appropriate_particle(),
                        title=self.context.get('title', '')
                    )
                elif text.lower().startswith(('goodbye', 'bye')):
                    result = pattern['farewell'].format(
                        particle=self._get_appropriate_particle()
                    )
            
            # Apply honorifics
            if self.context['relationship'] in self.honorifics:
                honorific = self.honorifics[self.context['relationship']]
                result = honorific['person'] + result if translated[0] == honorific['person'] else result
            
            result = self._post_process_translation(result, target_lang)
            
            # Apply context-specific rules
            result = self._apply_context_rules(result, target_lang)
            
            return {'translation': result, 'score': 0.6}
    
    def _post_process_translation(self, text: str, target_lang: str) -> str:
        """Post-process translation based on target language and context"""
        if target_lang == 'th':
            # Normalize Thai text
            if PYTHAINLP_AVAILABLE:
                text = normalize(text)
            
            # Add particles based on context
            if self.context['formality'] != 'informal' and not any(p in text for p in self.particles):
                particle = 'ครับ' if self.context['gender'] == 'male' else 'ค่ะ' if self.context['gender'] == 'female' else ''
                if particle:
                    text = f"{text}{particle}"
        else:
            # Capitalize first letter of sentences
            text = '. '.join(s.capitalize() for s in text.split('. '))
            
            # Ensure proper spacing
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Add final punctuation if missing
            if text and text[-1] not in '.!?':
                text += '.'
        
        return text

    def _get_appropriate_particle(self) -> str:
        """Get appropriate particle based on context"""
        if self.context['formality'] == 'formal':
            return 'ครับ' if self.context['gender'] == 'male' else 'ค่ะ'
        elif self.context['formality'] == 'informal':
            return 'จ้า' if self.context['gender'] == 'female' else 'จ้ะ'
        return 'ครับ' if self.context['gender'] == 'male' else 'ค่ะ'

    def _apply_context_rules(self, text: str, target_lang: str) -> str:
        """Apply context-specific rules to the translation"""
        if target_lang == 'th':
            # Apply dialect patterns
            if self.context['dialect'] != 'central':
                text = self._apply_dialect_patterns(text)
            
            # Apply domain-specific patterns
            if self.context['domain'] != 'general':
                text = self._apply_domain_patterns(text)
            
            # Apply age-specific patterns
            text = self._apply_age_patterns(text)
            
            # Apply register patterns
            text = self._apply_register_patterns(text)
            
            # Apply honorific level
            text = self._apply_honorific_level(text)
            
        return text
    
    def _apply_dialect_patterns(self, text: str) -> str:
        """Apply dialect-specific patterns"""
        dialect = self.context['dialect']
        if dialect in self.dialect_patterns:
            patterns = self.dialect_patterns[dialect]
            
            # Replace pronouns
            for eng, thai in patterns['pronouns'].items():
                text = text.replace(self.en_to_th[eng], thai)
            
            # Replace verbs
            for eng, thai in patterns['verbs'].items():
                text = text.replace(self.en_to_th[eng], thai)
            
            # Add dialect-specific particles
            if not any(p in text for p in patterns['particles']):
                text += patterns['particles'][0]
        
        return text
    
    def _apply_domain_patterns(self, text: str) -> str:
        """Apply domain-specific patterns"""
        domain = self.context['domain']
        if domain in self.domain_patterns:
            patterns = self.domain_patterns[domain]
            
            # Add domain-specific prefix
            if not any(p in text for p in patterns['prefix']):
                text = f"{patterns['prefix'][0]} {text}"
            
            # Replace verbs with domain-specific ones
            for verb in patterns['verbs']:
                if verb in text:
                    text = text.replace(verb, patterns['verbs'][verb])
        
        return text
    
    def _apply_age_patterns(self, text: str) -> str:
        """Apply age-specific patterns"""
        age = self.context['age_group']
        if age in self.age_patterns:
            patterns = self.age_patterns[age]
            
            # Adjust sentence length
            if patterns['sentence_length'] == 'short':
                sentences = self.sent_tokenizer(text) if PYTHAINLP_AVAILABLE else text.split('.')
                text = '. '.join(s.split(',')[0] for s in sentences)
            
            # Add age-appropriate particles
            if not any(p in text for p in patterns['particles']):
                text += patterns['particles'][0]
        
        return text
    
    def _apply_register_patterns(self, text: str) -> str:
        """Apply register-specific patterns"""
        register = self.context['register']
        if register in self.register_patterns:
            patterns = self.register_patterns[register]
            
            # Add register-appropriate particles
            if not any(p in text for p in patterns['particles']):
                text += patterns['particles'][0]
            
            # Replace conjunctions
            for conj in patterns['conjunctions']:
                if conj in text:
                    text = text.replace(conj, patterns['conjunctions'][conj])
        
        return text
    
    def _apply_honorific_level(self, text: str) -> str:
        """Apply honorific level patterns"""
        level = self.context['honorific_level']
        if level in self.honorific_levels:
            patterns = self.honorific_levels[level]
            
            # Replace pronouns
            for eng, thai in patterns['pronouns'].items():
                text = text.replace(self.en_to_th[eng], thai)
            
            # Replace verbs
            for eng, thai in patterns['verbs'].items():
                text = text.replace(self.en_to_th[eng], thai)
            
            # Add appropriate particles
            if not any(p in text for p in patterns['particles']):
                text += patterns['particles'][0]
        
        return text

def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en", engine: str = "default", context: Optional[Dict] = None) -> str:
    """
    Enhanced convenient function to translate text with context
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language ('th', 'en', or 'auto')
        target_lang (str): Target language ('th' or 'en')
        engine (str): Translation engine to use
        context (Optional[Dict]): Translation context
        
    Returns:
        str: Translated text
    """
    translator = ThaiTranslator(engine=engine)
    result = translator.translate(text, source_lang=source_lang, target_lang=target_lang, context=context)
    return result['translation']

def detect_language(text: str) -> str:
    """
    Detect if text is Thai or English
    
    Args:
        text (str): Text to detect
        
    Returns:
        str: Detected language code ('th' or 'en')
    """
    translator = ThaiTranslator()
    return translator.detect_language(text) 