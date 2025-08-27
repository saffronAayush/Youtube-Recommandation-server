from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from YoutubeRecommendation import get_video_embedding , get_similar_videos

app = FastAPI()

# ------------------
# MongoDB Connection
# ------------------
MONGO_URI = "mongodb+srv://aayushit6969:anshmishralovesaditrajdhvishrivastva@youtubeproj.gki5s.mongodb.net/"  # change if using Atlas
client = AsyncIOMotorClient(MONGO_URI)
db = client["youtube"]  
videos_collection = db["videos"]

# ------------------
# Models
# ------------------
class Video(BaseModel):
    title: str
    description: str
    category: str
    channelName: str
    embedding: list[float] = []  # list of floats
    similarVideos: list[dict] = []

# ------------------
# API Endpoints
# ------------------

@app.get("/")
async def home():
    
    return {"message": "Home Route is working in python server"}
    
@app.put("/create-embedding/{video_id}")
async def create_embedding(video_id: str):
    print("in creating embeddings")
    try:
        oid = ObjectId(video_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video_id format")
    
    video = await videos_collection.find_one({"_id": oid})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    embedding = get_video_embedding(video=video)

    await videos_collection.update_one(
    {"_id": oid},   # filter
    {"$set": {"embedding": embedding}}  # update
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    print("done creating embding")
    return {"success":True}

@app.put("/update-similar-video-matrix/{video_id}")
async def update_matrix(video_id:str):
    print("in update similarity")
    try:
        oid = ObjectId(video_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid video_id format")
    video = await videos_collection.find_one({"_id":oid},{"_id": 1, "embedding": 1})

    if not video or not video.get("embedding"):
        raise HTTPException(status_code=404, detail="Video not found or no embedding")

    all_videos_cursor = videos_collection.find(
        {}, {"_id": 1, "embedding": 1, "similarVideos": 1}
    )
    all_videos = await all_videos_cursor.to_list(length=None)

    similarities = get_similar_videos(video, all_videos)
    
    await videos_collection.update_one(
        {"_id": oid},
        {"$set": {"similarVideos": similarities}}
    )
    print(len(all_videos))
    for v in all_videos:
        print("in loop")
        if v["_id"] == video["_id"] :
            continue

        # find the score between this video and v
        score = next((s["score"] for s in similarities if str(s["videoId"]) == str(v["_id"])), None)
        print("scroe ",score)
        if score is None:
            continue

        # prepare the new entry
        new_entry = {"videoId": str(video["_id"]), "score": score}
        print("new entry")
        # get current similarVideos (may be empty)
        sim_list = v.get("similarVideos", [])
        print("sim list",sim_list)
        # insert at correct place (binary search style)
        left, right = 0, len(sim_list)
        while left < right:
            print("in while")
            mid = (left + right) // 2
            if sim_list[mid]["score"] < score:
                right = mid
            else:
                left = mid + 1
        print("left ",left)
        sim_list.insert(left, new_entry)
        print(sim_list)
        # update DB
        await videos_collection.update_one(
            {"_id": v["_id"]},
            {"$set": {"similarVideos": sim_list}}
        )
    print("done similarity")
    return {"success":True}










