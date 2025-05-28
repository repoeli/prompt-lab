# smoke_test.py - Professional OpenAI API Key Validation
import os
import sys
import time
from datetime import datetime
import openai
from dotenv import load_dotenv

def smoke_test_openai():
    """
    Professional smoke test for OpenAI API key validation.
    Tests key validity, response time, and basic functionality.
    """
    print("🧪 OpenAI API Smoke Test")
    print("=" * 50)
    
    # 1. Load environment variables
    load_dotenv()
    
    # 2. Validate API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid API key format (should start with 'sk-')")
        return False
    
    print(f"✅ API key found: {api_key[:7]}...")
    
    # 3. Initialize client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # 4. Test API call with timing
    try:
        print("🔄 Testing API call...")
        messages = [{"role": "user", "content": "Respond with exactly: 'API test successful'"}]
        
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=10,
            user="smoke-test"
        )
        end = time.perf_counter()
        
        # Extract response and metrics
        reply = response.choices[0].message.content
        latency_ms = (end - start) * 1000
        
        # Extract usage info
        usage = getattr(response, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        # Calculate estimated cost (gpt-4o-mini pricing)
        cost = (prompt_tokens * 0.60e-6) + (completion_tokens * 2.40e-6)
        
        print("✅ API call successful!")
        print(f"📝 Response: {reply}")
        print(f"⚡ Latency: {latency_ms:.1f} ms")
        print(f"🔢 Tokens: {prompt_tokens} → {completion_tokens} (total: {total_tokens})")
        print(f"💰 Cost: ${cost:.6f}")
        
        return True
        
    except openai.AuthenticationError:
        print("❌ Authentication failed - invalid API key")
        return False
    except openai.RateLimitError:
        print("❌ Rate limit exceeded - try again later")
        return False
    except openai.APIError as e:
        print(f"❌ OpenAI API error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main entry point for smoke test."""
    print(f"🚀 Prompt Lab Smoke Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = smoke_test_openai()
    
    print()
    if success:
        print("🎉 All tests passed! Your OpenAI setup is ready.")
        print("💡 You can now run your prompt lab experiments.")
        sys.exit(0)
    else:
        print("💥 Smoke test failed. Please fix the issues above.")
        print("📚 Check the README for setup instructions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
