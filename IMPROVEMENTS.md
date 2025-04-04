# Project Structure Improvements

## Files to Remove

1. Redundant Files:
- `thainlp/thai_spell_correction.py` (integrated into tokenizer)
- `thainlp/tag.py` (moved to classification)
- `thainlp/pos_tagging/hmm_tagger.py` (replaced by transformer models)
- `thainlp/feature_extraction/feature_extractor.py` (integrated into models)
- `thainlp/thai_data_augmentation.py` (can be added later if needed)
- `thainlp/model_hub/` (simplified for now)
- `thainlp/extensions/` (moved to utils)
- `thainlp/analytics/` (integrated into models)
- `thainlp/anomaly/` (not used)
- `thainlp/distributed/` (to be implemented later)
- `thainlp/docs/` (moved to project root)
- `thainlp/scaling/` (to be implemented later)
- `thainlp/security/` (not used)
- `thainlp/testing/` (moved to tests/)

## Files to Move

1. Models:
- Move token classification to `models/classification/`
- Move QA components to `models/qa/`
- Move text generation to `models/generation/`
- Move translation to `models/translation/`
- Move summarization to `models/summarization/`
- Move similarity to `models/similarity/`

2. Utils:
- Move monitoring to `utils/monitoring.py`
- Move Thai utilities to `utils/thai_utils.py`
- Move resource loading to `utils/resources.py`

3. Tests:
- Reorganize tests to mirror source structure
- Move all test files to appropriate subdirectories
- Update test imports

## New Files Created

1. Core Components:
- `thainlp/core/transformers.py`
- `thainlp/core/base.py`
- `thainlp/core/utils.py`

2. Models:
- `thainlp/models/classification/token_classifier.py`
- `thainlp/models/qa/text_qa.py`
- `thainlp/models/qa/table_qa.py`
- `thainlp/models/generation/text_generator.py`
- `thainlp/models/generation/fill_mask.py`
- `thainlp/models/translation/translator.py`
- `thainlp/models/summarization/summarizer.py`
- `thainlp/models/similarity/sentence_similarity.py`

3. Tokenization:
- `thainlp/tokenization/tokenizer.py`
- `thainlp/tokenization/maximum_matching.py`

4. Utils:
- `thainlp/utils/monitoring.py`
- `thainlp/utils/thai_utils.py`

5. Pipeline:
- `thainlp/pipelines/pipeline.py`

## Directory Structure

```
ChujaiThainlp/
├── thainlp/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── transformers.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── [model subdirectories]
│   ├── tokenization/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── maximum_matching.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── monitoring.py
│   │   └── thai_utils.py
│   └── pipelines/
│       ├── __init__.py
│       ├── base.py
│       └── pipeline.py
└── tests/
    ├── __init__.py
    └── [test subdirectories]
```

## Benefits of New Structure

1. Better Organization:
- Clear separation of concerns
- Logical grouping of components
- Easier to maintain and extend

2. Improved Maintainability:
- Less code duplication
- Clear dependencies
- Better test organization

3. Better User Experience:
- Simpler import paths
- Consistent interfaces
- Clear documentation

4. Reduced Complexity:
- Fewer files to maintain
- Clearer responsibility boundaries
- Easier to understand

## Next Steps

1. Create directory structure
2. Move files to new locations
3. Update imports in all files
4. Remove redundant files
5. Update documentation
6. Run and fix tests
7. Update version number
8. Create new release
