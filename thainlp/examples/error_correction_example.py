"""
Example usage of Thai Error Correction Agent
"""
import os
import sys

def run_examples():
    from thainlp.agent.error_correction_agent import ThaiErrorCorrectionAgent
    
    # Initialize error correction agent
    agent = ThaiErrorCorrectionAgent()
    
    # Example 1: Wrong keyboard layout
    print("\nExample 1: Wrong keyboard layout")
    text1 = "l;ylfu"  # พิมพ์คำว่า "สวัสดี" ผิดแป้นพิมพ์
    result1 = agent.correct_text(text1)
    print(f"Original: {text1}")
    print(f"Corrected: {result1['corrected']}")
    print(f"Corrections: {result1['corrections']}")
    
    # Example 2: Social media style text
    print("\nExample 2: Social media style text")
    text2 = "เธอออออ คิดถึงงงง มากกกกก"
    result2 = agent.correct_text(text2, aggressive=True)
    print(f"Original: {text2}")
    print(f"Corrected: {result2['corrected']}")
    print(f"Corrections: {result2['corrections']}")
    
    # Example 3: Common Thai typos
    print("\nExample 3: Common Thai typos")
    text3 = "เเปนเรือง ทีเกิดขึน"  # "เป็นเรื่อง ที่เกิดขึ้น"
    result3 = agent.correct_text(text3)
    print(f"Original: {text3}")
    print(f"Corrected: {result3['corrected']}")
    print(f"Corrections: {result3['corrections']}")
    
    # Example 4: Mixed errors
    print("\nExample 4: Mixed errors")
    text4 = "k;k เเละะะ งัยยย"  # "ค่ะ และ ไง"
    result4 = agent.correct_text(text4, aggressive=True)
    print(f"Original: {text4}")
    print(f"Corrected: {result4['corrected']}")
    print(f"Corrections: {result4['corrections']}")
    
    # Example 5: Error analysis
    print("\nExample 5: Error analysis")
    text5 = "helloooo สบายดีีีี มากกก"
    analysis = agent.analyze_errors(text5)
    print(f"Text: {text5}")
    print(f"Analysis: {analysis}")
    print(f"Suggestions: {analysis['suggestions']}")
    
    # Example 6: Correction suggestions
    print("\nExample 6: Correction suggestions")
    text6 = "เคา เปน คนที่ ชอบบ มากกก"
    suggestions = agent.suggest_corrections(text6)
    print(f"Text: {text6}")
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion['word']}: {suggestion.get('suggestion') or suggestion.get('suggestions')}")

if __name__ == "__main__":
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    
    print("Running Thai Error Correction examples...")
    print("Project root:", project_root)
    
    try:
        run_examples()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)