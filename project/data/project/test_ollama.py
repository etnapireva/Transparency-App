import ollama

response = ollama.chat(
    model="gemma3:1b",
    messages=[{"role": "user", "content": "Hello, si je?"}]
)

# Printo tekstin e përgjigjes
print(response.message.content)
