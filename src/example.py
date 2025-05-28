# example.py
from runner import run_chat

if __name__ == "__main__":
    messages = [{"role": "user", "content": "Hello, runner!"}]
    reply = run_chat(
        model="gpt-4o-mini",
        messages=messages,
        user="example-user"
    )
    print("Assistant replied:", reply)
