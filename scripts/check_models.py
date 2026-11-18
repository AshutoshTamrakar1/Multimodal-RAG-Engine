import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
print("=" * 60)
print(f"🔑 Testing API Key: {api_key[:8]}... (Length: {len(api_key)})")
print("=" * 60)

if not api_key:
    print("❌ Error: GOOGLE_API_KEY is not set in .env")
    exit(1)

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    
    print("\n📡 Querying Google API for models supported by your key...")
    models = list(client.models.list())
    
    print(f"\n✅ SUCCESS! Found {len(models)} accessible models:\n")
    generate_models = []
    embed_models = []

    for m in models:
        model_name = getattr(m, "name", str(m))
        actions = getattr(m, "supported_actions", [])
        if "embedContent" in actions:
            embed_models.append(model_name)
        else:
            generate_models.append(model_name)

    print("--- 📝 Text / Vision Models ---")
    for name in generate_models[:10]:
        print(f"  • {name}")
    if len(generate_models) > 10:
        print(f"  ... and {len(generate_models) - 10} more.")

    print("\n--- 🔮 Embedding Models ---")
    for name in embed_models:
        print(f"  • {name}")

except Exception as e:
    print(f"\n❌ Error querying models: {e}")
    print("\nDebugging tips:")
    print("1. If 403 Forbidden: The Generative Language API may need to be enabled for this project.")
    print("2. If 404 or 400: Check if project or key has region/service restrictions.")
