# smoke_test.py
import os
import time
import openai

# 1. Ensure your key is loaded (from env, dotenv, etc.)
#    e.g. export OPENAI_API_KEY=sk-…
#    or have python-dotenv load it for you.

# 2. Initialize the client
client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

# 3. Build and send the request, timing it
messages = [{"role": "user", "content": "Ping"}]
start = time.perf_counter()
res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    user="smoke-test",
)
end = time.perf_counter() # (you can also use time.time() if you prefer)
# Note: If you want to use a different model, change "gpt-4o-mini" to your desired model.

# 4. Extract the assistant reply
reply = res.choices[0].message.content

# 5. Extract usage info (requires openai>=1.14+)
usage = getattr(res, "usage", None)
prompt_tokens = usage.prompt_tokens if usage else None
completion_tokens = usage.completion_tokens if usage else None
total_tokens = usage.total_tokens if usage else None

# 6. Print everything
print(f"\nReply: {reply!r}")
print(f"Latency: {(end - start)*1000:.1f} ms")
if usage:
    print(f"Prompt tokens:      {prompt_tokens}")
    print(f"Completion tokens:  {completion_tokens}")
    print(f"Total tokens:       {total_tokens}")
    # (you can multiply total_tokens by your per-1k-token rate to estimate cost)
else:
    print("No usage info available on this client version.")
