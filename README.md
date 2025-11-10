
#  Vibe Matcher


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

## 🧩 Project Structure

---
vibe_matcher/
│
├── .env                     # Stores your OpenRouter API key
├── requirements.txt         # All required dependencies
├── vibe_matcher.py          # Main Python script
├── vibe_matcher.log         # Auto-generated log file
├── vibe_matcher_output.csv  # Output file with vibe match results
└── README.md                # Project documentation

