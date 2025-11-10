import os
import time
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# 1️. Logging Setup

logging.basicConfig(
    filename="vibe_matcher.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info(" Vibe Matcher script started.")


# 2️. Load API Key

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    logging.error(" OPENROUTER_API_KEY not found in .env file.")
    raise ValueError(" OPENROUTER_API_KEY not found in .env file.")
else:
    logging.info(" API key loaded successfully.")

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)
logging.info(" OpenRouter client initialized successfully.")


# 3️. Mock Product Data

data = [
    {"name": "Boho Dress", "desc": "Flowy, earthy tones for festival vibes.", "vibes": ["boho", "relaxed"]},
    {"name": "Street Jacket", "desc": "Bold patterns and edgy design for city life.", "vibes": ["urban", "chic"]},
    {"name": "Cozy Sweater", "desc": "Warm, soft knit perfect for a relaxed evening.", "vibes": ["cozy", "casual"]},
    {"name": "Sporty Sneakers", "desc": "Lightweight sneakers for an active, energetic look.", "vibes": ["sporty", "energetic"]},
    {"name": "Classic Blazer", "desc": "Tailored fit for a professional and elegant vibe.", "vibes": ["formal", "classic"]},
    {"name": "Denim Jeans", "desc": "Casual blue jeans for everyday comfort and style.", "vibes": ["casual", "urban"]},
    {"name": "Floral Skirt", "desc": "Bright floral print for cheerful summer days.", "vibes": ["boho", "vibrant"]},
    {"name": "Leather Boots", "desc": "Rugged and stylish boots for a confident, bold look.", "vibes": ["bold", "urban"]},
]
df = pd.DataFrame(data)
logging.info(" Mock product dataset created successfully.")


# 4️. Embedding Function

def get_embedding(text):
    """Generate text embeddings using OpenRouter API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"⚠️ Error generating embedding for '{text}': {e}")
        return None

# Generate embeddings
print("🔹 Generating embeddings for products... (may take a few seconds)")
logging.info("🔹 Generating embeddings for products...")
df["embedding"] = df["desc"].apply(get_embedding)
df = df.dropna(subset=["embedding"]).reset_index(drop=True)
logging.info(f" Generated embeddings for {len(df)} products.")


# 5️ Vibe Matching Function

def match_vibe(query, top_k=3):
    """Find top matching products for a vibe query"""
    print(f"\n Searching for vibe: '{query}'")
    logging.info(f" Searching for vibe query: '{query}'")
    start = time.time()

    query_embed = get_embedding(query)
    if query_embed is None:
        logging.error(f" Failed to generate embedding for query: '{query}'")
        return None

    # Compute cosine similarity
    similarities = cosine_similarity([query_embed], df["embedding"].tolist())[0]
    df["score"] = similarities

    top_matches = df.sort_values(by="score", ascending=False).head(top_k)
    end = time.time()
    elapsed = round(end - start, 2)

    print(f"\nTop {top_k} Matches for '{query}':")
    for i, row in top_matches.iterrows():
        print(f" {i+1}. {row['name']} — {row['desc']} (Score: {row['score']:.3f})")

    print(f"⏱ Search completed in {elapsed}s")

    logging.info(f" Query '{query}' completed in {elapsed}s. Top match: {top_matches.iloc[0]['name']}")
    
    # Save results to log CSV
    results = top_matches[["name", "desc", "score"]].copy()
    results["query"] = query
    results["time_taken(s)"] = elapsed
    results.to_csv("vibe_matcher_output.csv", mode="a", header=not os.path.exists("vibe_matcher_output.csv"), index=False)

    return top_matches


# 6️ Run Test Queries

queries = ["energetic urban chic", "relaxed cozy vibe", "professional office look"]

for q in queries:
    match_vibe(q)

print("\n All queries completed successfully!")
logging.info(" All vibe queries completed successfully. Results saved to vibe_matcher_output.csv.")
