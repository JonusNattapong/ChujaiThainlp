import thainlp
print("Available attributes:", dir(thainlp))
text = "ร้านอาหารนี้อร่อยมาก บรรยากาศดี แนะนำให้มาลอง"
try:
    result = thainlp.classify_text(text, category="sentiment")
    print("Classification result:", result)
except Exception as e:
    print("Error:", e)
    print("Module info:", thainlp)
