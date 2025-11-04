import os
os.environ["OPENAI_API_KEY"] = "sk-test-key-12345"

from app.config import settings
print("✅ Config loaded!")
print(f"API Key: {settings.openai_api_key[:20]}...")
print(f"Model: {settings.openai_model}")