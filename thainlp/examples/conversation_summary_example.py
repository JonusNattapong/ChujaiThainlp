"""Example conversation for summarization"""

EXAMPLE_CONVERSATION = """
    สมชาย: สวัสดีครับคุณสมหญิง เรามาคุยเรื่องการพัฒนา NLP สำหรับภาษาไทยกันดีกว่า
    สมหญิง: สวัสดีค่ะ ดิฉันคิดว่าเราควรเริ่มจากการทำความเข้าใจพื้นฐานก่อน
    สมชาย: ผมเห็นด้วยครับ การตัดคำเป็นพื้นฐานที่สำคัญ แล้วเราจะใช้ไลบรารีอะไรดีครับ?
    สมหญิง: ดิฉันคิดว่าเราควรใช้ ThaiNLP เป็นพื้นฐาน แล้วพัฒนาต่อยอด เพราะมันมีฟังก์ชันพื้นฐานครบถ้วน
    สมชาย: เห็นด้วยครับ แล้วเรามีทีมงานกี่คนที่จะช่วยกันพัฒนาโปรเจกต์นี้?
    สมหญิง: ตอนนี้เรามีทีมงานหลัก 5 คน และมีนักพัฒนาที่สนใจจะร่วมโปรเจกต์อีกประมาณ 10 คนค่ะ
    สมชาย: ดีมากครับ ถ้างั้นเรามาวางแผนการพัฒนากันเลยดีไหมครับ?
    สมหญิง: ได้เลยค่ะ ดิฉันคิดว่าเราควรแบ่งงานเป็นโมดูลๆ เพื่อให้ทุกคนสามารถทำงานคู่ขนานกันได้
    สมชาย: ครับ ผมจะรับผิดชอบส่วนการตัดคำและวิเคราะห์ไวยากรณ์
    สมหญิง: ดิฉันจะดูแลส่วนการประมวลผลความหมายและการแปลภาษาค่ะ
"""

def get_example_conversation():
    """Get example conversation text"""
    return EXAMPLE_CONVERSATION

def summarize_example():
    """Example of using conversation summarizer"""
    from ..summarization.conversation_summarizer import ConversationSummarizer
    
    summarizer = ConversationSummarizer()
    summary = summarizer.summarize(EXAMPLE_CONVERSATION)
    
    print("Original Conversation:")
    print(EXAMPLE_CONVERSATION)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    summarize_example()
