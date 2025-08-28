from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # lightest one
    return _model

def get_video_embedding(video):
    model = get_model()
    text =  f"{video['title']} {video['description']} {video['category']} {video['channelName']}"
    emb = model.encode([text])[0]
    return emb.tolist()

def get_search_embedding(search_text):
    model = get_model()
    text = search_text
    embedding = model.encode([text])[0]
    return embedding.tolist()


def get_similar_videos(video, all_videos):
    base_emb = np.array(video["embedding"]).reshape(1, -1)
    similarities = []
    for v in all_videos:
        if v["_id"] == video["_id"]:
            continue 
        if not v.get("embedding"):
            continue
        emb = np.array(v["embedding"]).reshape(1, -1)
        score = float(cosine_similarity(base_emb, emb)[0][0])
        similarities.append({"videoId": str(v["_id"]), "score": score})
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities


def get_similar_videos_from_embedding(embedding, all_videos):

    # Convert target embedding to numpy
    target_emb = np.array(embedding).reshape(1, -1)

    similarities = []

    for v in all_videos:
        if not v.get("embedding"):  # skip if no embedding
            continue

        emb = np.array(v["embedding"]).reshape(1, -1)
        sim = cosine_similarity(target_emb, emb).item()
        if sim>=0:
            similarities.append({"videoId": str(v["_id"]), "score": sim})

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities

