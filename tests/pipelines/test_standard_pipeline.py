"""
Tests for the Thai NLP pipeline
"""
import pytest
from thainlp.pipelines.thai_nlp_pipeline import ThaiNLPPipeline

@pytest.fixture
def pipeline():
    """Create pipeline instance for testing"""
    return ThaiNLPPipeline(
        components=[
            'classification',
            'qa',
            'translation',
            'generation',
            'fill_mask',
            'summarization',
            'similarity'
        ],
        device='cpu',  # Use CPU for testing
        batch_size=2    # Small batch size for testing
    )

def test_pipeline_initialization():
    """Test pipeline component initialization"""
    # Test with specific components
    pipeline = ThaiNLPPipeline(
        components=['translation', 'summarization'],
        device='cpu'
    )
    assert 'translation' in pipeline.components
    assert 'summarization' in pipeline.components
    assert 'qa' not in pipeline.components
    
    # Test with no components specified (should initialize all)
    pipeline = ThaiNLPPipeline(device='cpu')
    assert 'classification' in pipeline.components
    assert 'qa' in pipeline.components
    assert 'translation' in pipeline.components
    assert 'generation' in pipeline.components
    assert 'fill_mask' in pipeline.components
    assert 'summarization' in pipeline.components
    assert 'similarity' in pipeline.components

def test_text_analysis(pipeline):
    """Test basic text analysis functionality"""
    text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"
    
    results = pipeline.analyze(
        text,
        tasks=['tokens', 'entities', 'translation', 'summary']
    )
    
    assert 'tokens' in results
    assert 'entities' in results
    assert 'translation' in results
    assert 'summary' in results
    
    # Check translation
    assert isinstance(results['translation']['en'], str)
    assert len(results['translation']['en']) > 0
    
    # Check tokens
    assert len(results['tokens']) > 0
    assert 'token' in results['tokens'][0]
    assert 'pos' in results['tokens'][0]

def test_question_answering(pipeline):
    """Test question answering functionality"""
    context = """
    ประเทศไทยมีประชากรประมาณ 70 ล้านคน มีกรุงเทพมหานครเป็นเมืองหลวง
    """
    question = "เมืองหลวงของประเทศไทยคือที่ไหน"
    
    answer = pipeline.answer_question(
        question=question,
        context=context,
        return_scores=True
    )
    
    assert 'answer' in answer
    assert 'score' in answer
    assert isinstance(answer['answer'], str)
    assert isinstance(answer['score'], float)
    assert 'กรุงเทพ' in answer['answer'].lower()

def test_translation(pipeline):
    """Test translation functionality"""
    thai_text = "สวัสดีครับ"
    english_text = "Hello"
    
    # Thai to English
    en_translation = pipeline.translate(
        text=thai_text,
        source_lang='th',
        target_lang='en'
    )
    assert isinstance(en_translation, str)
    assert len(en_translation) > 0
    
    # English to Thai
    th_translation = pipeline.translate(
        text=english_text,
        source_lang='en',
        target_lang='th'
    )
    assert isinstance(th_translation, str)
    assert len(th_translation) > 0
    
    # Batch translation
    texts = [thai_text, thai_text]
    translations = pipeline.translate(
        text=texts,
        source_lang='th',
        target_lang='en'
    )
    assert isinstance(translations, list)
    assert len(translations) == 2

def test_text_generation(pipeline):
    """Test text generation functionality"""
    prompt = "เทคโนโลยีปัญญาประดิษฐ์"
    
    # Single sequence
    generated = pipeline.generate_text(
        prompt=prompt,
        max_length=50,
        num_return_sequences=1
    )
    assert isinstance(generated, str)
    assert len(generated) > 0
    
    # Multiple sequences
    generated = pipeline.generate_text(
        prompt=prompt,
        max_length=50,
        num_return_sequences=2
    )
    assert isinstance(generated, list)
    assert len(generated) == 2

def test_summarization(pipeline):
    """Test summarization functionality"""
    text = """
    กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย 
    เป็นศูนย์กลางการปกครอง การศึกษา การคมนาคมขนส่ง การเงินการธนาคาร การพาณิชย์ 
    การสื่อสาร และความเจริญของประเทศ
    """
    
    # Single text
    summary = pipeline.summarize(
        text=text,
        ratio=0.3
    )
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(text)
    
    # Batch summarization
    texts = [text, text]
    summaries = pipeline.summarize(
        text=texts,
        ratio=0.3
    )
    assert isinstance(summaries, list)
    assert len(summaries) == 2

def test_similarity(pipeline):
    """Test similarity functionality"""
    text1 = "ร้านอาหารนี้อร่อยมาก"
    text2 = "อาหารที่ร้านนี้รสชาติดีมาก"
    text3 = "วันนี้อากาศร้อนมาก"
    
    # Calculate similarity scores
    score1 = pipeline.calculate_similarity(text1, text2)
    score2 = pipeline.calculate_similarity(text1, text3)
    
    assert isinstance(score1, float)
    assert isinstance(score2, float)
    assert 0 <= score1 <= 1
    assert 0 <= score2 <= 1
    assert score1 > score2  # Similar texts should have higher score
    
    # Test finding similar texts
    documents = [text1, text2, text3]
    results = pipeline.find_similar_texts(
        query=text1,
        candidates=documents,
        top_k=2
    )
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][1], float)

def test_fill_mask(pipeline):
    """Test mask filling functionality"""
    text = "ประเทศไทยมี[MASK]ที่หลากหลาย"
    
    # Single prediction
    predictions = pipeline.fill_mask(
        text=text,
        top_k=1
    )
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert 'token' in predictions[0]
    assert 'score' in predictions[0]
    
    # Multiple predictions
    predictions = pipeline.fill_mask(
        text=text,
        top_k=3
    )
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert all('token' in p for p in predictions)
    assert all('score' in p for p in predictions)

def test_error_handling(pipeline):
    """Test error handling"""
    # Test invalid component access
    with pytest.raises(ValueError):
        pipeline.get_component('invalid_component')
    
    # Test missing required component
    temp_pipeline = ThaiNLPPipeline(
        components=['translation'],
        device='cpu'
    )
    with pytest.raises(ValueError):
        temp_pipeline.answer_question("question", "context")
        
    # Test invalid language codes
    with pytest.raises(ValueError):
        pipeline.translate("text", source_lang='invalid', target_lang='en')
