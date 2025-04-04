# Changelog

## [2.0.0] - 2025-04-04

Major reorganization and enhancement of the library with improved transformer-based models and better structure.

### Added

- New unified `ThaiNLPPipeline` for integrated access to all components
- Advanced transformer models:
  - XLM-RoBERTa for token classification and NER
  - M2M100 for neural machine translation
  - BART for abstractive summarization
  - XGLM for text generation
  - MPNet for sentence embeddings and similarity
  - TAPAS for table question answering

### Enhanced

#### Core Architecture

- Complete codebase restructuring for better organization
- Unified tokenization system supporting both Thai and English
- Efficient batch processing across all components
- Progress monitoring and resource tracking
- Better error handling and logging

#### Tokenization

- New unified `ThaiTokenizer` class
- Improved maximum matching algorithm
- Better handling of Thai-English mixed text
- Support for word and subword tokenization
- Efficient caching mechanisms

#### Models

- Token Classification:
  - Enhanced NER capabilities
  - Support for custom label sets
  - Confidence scoring
  - Fine-tuning support

- Question Answering:
  - Both text and table QA support
  - Improved context handling
  - Multiple answer generation
  - Better answer ranking

- Translation:
  - Neural machine translation
  - Quality metrics
  - Custom vocabulary support
  - Batch translation

- Text Generation:
  - Advanced transformer-based generation
  - Improved streaming support
  - Better control parameters
  - Prompt templates

- Summarization:
  - Both extractive and abstractive methods
  - ROUGE metrics integration
  - Length control
  - Efficient batch processing

- Sentence Similarity:
  - Multiple similarity metrics
  - Cross-encoder support
  - Efficient similarity search
  - Improved ranking

#### Infrastructure

- Better dependency management
- Comprehensive test suite
- Improved documentation
- Example notebooks
- Development utilities

### Changed

- Updated all base models to transformer architectures
- Standardized API interfaces across components
- Improved resource management
- Enhanced error handling
- Updated documentation format

### Removed

- Legacy n-gram based generation
- Basic dictionary-based translation
- Simple similarity metrics
- Redundant utility functions
- Deprecated components

### Fixed

- Memory leaks in batch processing
- Thread safety issues
- Resource cleanup
- Import conflicts
- Documentation inconsistencies

### Performance

- Improved batch processing efficiency
- Better memory management
- Reduced CPU/GPU switching
- Optimized tokenization
- Faster similarity search

### Security

- Added input validation
- Improved error handling
- Resource usage limits
- Timeout controls

## [1.0.0] - 2024-01-01

Initial release with basic Thai NLP functionality:

- Basic text classification
- Rule-based tokenization
- Dictionary-based translation
- Simple NER
- Basic text utilities
