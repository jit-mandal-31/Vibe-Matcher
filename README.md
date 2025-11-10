
#  Vibe-Matcher


---

**Vibe Matcher** is an AI-powered fashion recommendation prototype that matches clothing or product items to user *“vibes”* using **OpenRouter’s Embedding API**.  
It analyzes product descriptions and finds the closest matches based on semantic similarity — powered by embeddings and cosine similarity.

---

## 🚀 Features

- 🧠 AI-based vibe matching using **text-embedding-3-small**
- 📊 Semantic similarity with **cosine similarity**
- 🧾 Automatic logging of all activities in `vibe_matcher.log`
- 💾 Results saved to `vibe_matcher_output.csv`
- ⚙️ Built using **Python, Pandas, Scikit-learn, dotenv, and OpenAI SDK**

---

## ⚙️ Installation & Setup

---

### 1. Install Dependences

pip install -r requirements.txt

### 2. Create a .env File

OPENROUTER_API_KEY= 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

OPENAI_API_BASE = https://openrouter.ai/api/v1

### 3. Run the Script

---

## Output

---


python vibe_matcher.py

