# Changelog

All notable changes to the ChujaiThaiNLP project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-04-09

### Added
- **Thai Dialect Processing**: เพิ่มความสามารถในการประมวลผลภาษาไทยถิ่น
  - เพิ่มโมดูล `ThaiDialectProcessor` สำหรับตรวจจับและแปลภาษาไทยถิ่น
  - รองรับภาษาถิ่นเหนือ อีสาน ใต้ ภาษากลาง และมลายูปัตตานี
  - เพิ่ม `DialectTokenizer` สำหรับตัดคำภาษาไทยถิ่น
- **Multimodal Processing**: เพิ่มการประมวลผลแบบหลายโมดัล
  - เพิ่มฟังก์ชัน `transcribe_audio`, `caption_image` และ `answer_visual_question`
  - เพิ่มความสามารถในการประมวลผลเอกสารแบบซับซ้อนด้วย `process_multimodal`
- **Text-to-Speech**: เพิ่มความสามารถในการสังเคราะห์เสียงภาษาไทย

### Fixed
- แก้ไขปัญหาการ import `torch` ในโมดูล `multimodal/pipeline.py`
- แก้ไขปัญหาการ import `detect_regional_dialect` ในตัวอย่าง `dialect_example.py`
- ปรับปรุงการจัดการข้อผิดพลาดในฟังก์ชัน `speech_synthesis_with_dialect`

### Changed
- ปรับปรุงโครงสร้างโมดูลหลักให้มีความยืดหยุ่นมากขึ้น
- ปรับปรุงประสิทธิภาพของการตัดคำภาษาไทย
- ปรับปรุงคุณภาพการแปลภาษาไทยถิ่น

## [0.1.0] - 2025-03-15

### Added
- **Core NLP Capabilities**: ฟังก์ชันพื้นฐานสำหรับประมวลผลภาษาธรรมชาติภาษาไทย
  - word_tokenize: การตัดคำภาษาไทย
  - get_entities: การรู้จำชื่อเฉพาะ
  - get_sentiment: การวิเคราะห์ความรู้สึก
  - generate: การสร้างข้อความ
- **Initial Model Support**: รองรับโมเดลภาษาไทยเริ่มต้น
  - Transformer-based models สำหรับงาน NLP ภาษาไทย
  - เพิ่มระบบโหลดโมเดลอัตโนมัติ
- **Documentation**: เพิ่มเอกสารประกอบการใช้งานเบื้องต้น

[0.2.0]: https://github.com/JonusNattapong/ChujaiThainlp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JonusNattapong/ChujaiThainlp/releases/tag/v0.1.0