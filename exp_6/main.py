# ==============================
# YouTube Comment Sentiment API
# ==============================

from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -----------------------------------
# STEP 1: Initialize FastAPI App
# -----------------------------------

app = FastAPI()

# -----------------------------------
# STEP 2: Add Your YouTube API Key
# -----------------------------------

YOUTUBE_API_KEY = "AIzaSyCKdX8Wgk9eCh3DNE52Io9SdgVnjPlYV5w"   # 🔴 Replace this

# -----------------------------------
# STEP 3: Initialize YouTube Client
# -----------------------------------

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# -----------------------------------
# STEP 4: Fetch YouTube Comments
# -----------------------------------

def fetch_comments(video_id: str, max_comments: int):

    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100)
        )

        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

    except HttpError as e:
        return {"error": str(e)}

    return comments

# -----------------------------------
# STEP 5: Sentiment Analysis Function
# -----------------------------------

def analyze_sentiment(text: str):

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity

# -----------------------------------
# STEP 6: Root Endpoint
# -----------------------------------

@app.get("/")
def root():
    return {"message": "YouTube Comment Sentiment API is running successfully!"}

# -----------------------------------
# STEP 7: Request Model
# -----------------------------------

class CommentRequest(BaseModel):
    video_id: str
    count: int

# -----------------------------------
# STEP 8: POST Endpoint
# -----------------------------------

@app.post("/fetch_comments")
def get_comments(request: CommentRequest):

    comments = fetch_comments(request.video_id, request.count)

    # If API error occurred
    if isinstance(comments, dict) and "error" in comments:
        return comments

    results = []

    for comment in comments:
        sentiment, polarity = analyze_sentiment(comment)

        results.append({
            "comment": comment,
            "sentiment": sentiment,
            "polarity": polarity
        })

    return {
        "video_id": request.video_id,
        "total_comments": len(results),
        "results": results
    }
