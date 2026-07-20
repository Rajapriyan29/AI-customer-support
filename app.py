from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import os
from dotenv import load_dotenv
import traceback
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is missing in .env file!")

client = genai.Client(api_key=GEMINI_API_KEY)

# Try models in this order (fallback)
MODEL_OPTIONS = [
    "gemini-3.5-flash",      # Fast & capable
    "gemini-2.5-pro",        # Stronger reasoning
    "gemini-3.1-flash-lite", # Lighter fallback
]

KNOWLEDGE_BASE = """
# Company Knowledge Base
## Shipping
- Standard: 3-7 business days (free over $50)
- Express: 1-3 days ($12.99)

## Returns
- 30-day return policy
- Must be unused with original packaging

## Support
- All products have 2-year warranty
"""

SYSTEM_PROMPT = f"""You are a friendly customer support assistant. Use the knowledge base above to answer questions.

{KNOWLEDGE_BASE}
"""

def generate_with_retry(contents, max_retries=3):
    for attempt in range(max_retries):
        for model_name in MODEL_OPTIONS:
            try:
                print(f"🔄 Trying model: {model_name} (attempt {attempt+1})")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                print(f"✅ Success with {model_name}")
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    print(f"⚠️  Model {model_name} overloaded, trying next...")
                    continue
                elif "429" in error_str:
                    print("⏳ Rate limited, waiting...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e  # Re-raise other errors
        time.sleep(2 ** attempt)  # Exponential backoff
    raise Exception("All models unavailable after retries")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data['message']
        history = data.get('history', [])

        # Build conversation (same as before)
        contents = [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "model", "parts": [{"text": "Got it! I'll help using the company knowledge."}]}
        ]

        for msg in history[-10:]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        # Generate with retry + fallback
        bot_reply = generate_with_retry(contents)

        return jsonify({"response": bot_reply})

    except Exception as e:
        print("🔥 ERROR in /chat route:")
        print(traceback.format_exc())
        return jsonify({
            "error": "Service is temporarily busy. Please try again in a moment."
        }), 503


@app.route('/')
def index():
    return open('index.html', encoding='utf-8').read()


if __name__ == '__main__':
    print("🚀 Starting SupportBot on http://localhost:5000")
    app.run(debug=True, port=5000)