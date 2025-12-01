"""
Quick test to verify Gemini API key works
"""
import os
import sys

print(f"Python version: {sys.version}")

# Load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found!")
    exit(1)

print(f"✅ Found API key: {api_key[:8]}...{api_key[-4:]}")

# Try importing google.generativeai
try:
    import google.generativeai as genai
    print(f"✅ google-generativeai version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
    
    genai.configure(api_key=api_key)
    
    # Try the new API style
    model = genai.GenerativeModel('gemini-2.0-flash')  # Try older model name
    print("\n📡 Calling Gemini API...")
    response = model.generate_content("Say hello!")
    print(f"🤖 Response: {response.text.strip()}")
    print("\n✅ SUCCESS!")
    
except AttributeError as e:
    print(f"\n❌ Package issue: {e}")
    print("\n🔧 Try upgrading the package:")
    print("   pip install --upgrade google-generativeai")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
