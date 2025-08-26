from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model (only once, at startup)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_video_embedding(video):
   
    # Combine video metadata into one text document
    text = f"{video['title']} {video['description']} {video['category']} {video['channelName']}"
    
    # Generate embedding
    embedding = model.encode(text)
    
    # Convert numpy array to Python list (for MongoDB storage)
    return embedding.tolist()


def get_similar_videos(video, all_videos):

    # Convert target embedding to numpy
    target_emb = np.array(video["embedding"]).reshape(1, -1)

    similarities = []

    for v in all_videos:
        if v["_id"] == video["_id"]:
            continue  # skip the same video itself
        if not v.get("embedding"):  # skip if no embedding
            continue

        emb = np.array(v["embedding"]).reshape(1, -1)
        sim = cosine_similarity(target_emb, emb).item()
        similarities.append({"videoId": str(v["_id"]), "score": sim})

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities
