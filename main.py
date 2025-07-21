from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(__file__))
from game_logic import (
    init_game, get_clothes, make_pick, make_ranking, get_leaderboard, get_game_state
)
import base64
from transformers import AutoProcessor, CLIPModel
import torch
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the absolute path to the dataset based on the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLOTHES_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'cloth')
PEOPLE_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'image')

app.mount("/clothes", StaticFiles(directory=CLOTHES_DIR), name="clothes")
app.mount("/people", StaticFiles(directory=PEOPLE_DIR), name="people")

clip_model_name = "patrickjohncyh/fashion-clip"
processor = AutoProcessor.from_pretrained(clip_model_name)
model = CLIPModel.from_pretrained(clip_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

import os
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")  
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("fashion-clip-index") 

class StartGameRequest(BaseModel):
    num_players: int
    num_rounds: int
    timer: int
    n_clothes: Optional[int] = 10

class PickRequest(BaseModel):
    session_id: str
    round_num: int
    player: str
    picks: Dict[str, int]

class RankRequest(BaseModel):
    session_id: str
    round_num: int
    player: str
    ranking: List[int]

@app.post("/start_game")
def start_game(req: StartGameRequest):
    try:
        n_clothes = req.n_clothes if req.n_clothes is not None else 10
        session_id = init_game(req.num_players, req.num_rounds, req.timer, n_clothes)
        game_state = get_game_state(session_id)
        return {
            "session_id": session_id, 
            "players": game_state['players'],
            "timer": game_state['timer'],
            "num_rounds": game_state['num_rounds']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design")
def design(prompt: str = Body(...)):
    # Placeholder: return a static image for now
    return {"image": "https://images.pexels.com/photos/1884581/pexels-photo-1884581.jpeg?auto=compress&cs=tinysrgb&w=500"}

@app.get("/clothes_list")
def clothes_list(
    session_id: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    category: str = Query(None),
    color: str = Query(None),
    company: str = Query(None),
    priceMax: int = Query(None)
):
    try:
        if session_id == "store":
            clothes_dir = CLOTHES_DIR
            all_clothes = [f for f in os.listdir(clothes_dir) if f.endswith('.jpg')]
            # Filtering logic placeholder (no metadata available)
            # If you add metadata, filter here
            total = len(all_clothes)
            start = (page - 1) * page_size
            end = start + page_size
            page_clothes = all_clothes[start:end]
            return {"clothes": page_clothes, "total": total, "page": page, "page_size": page_size}
        elif session_id == "people":
            people_dir = PEOPLE_DIR
            all_people = [f for f in os.listdir(people_dir) if f.endswith('.jpg')]
            total = len(all_people)
            start = (page - 1) * page_size
            end = start + page_size
            page_people = all_people[start:end]
            return {"clothes": page_people, "total": total, "page": page, "page_size": page_size}
        else:
            clothes = get_clothes(session_id)
            total = len(clothes)
            start = (page - 1) * page_size
            end = start + page_size
            page_clothes = clothes[start:end]
            return {"clothes": page_clothes, "total": total, "page": page, "page_size": page_size}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/search_clothes")
def search_clothes(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100)
):

    inputs = processor(text=[query], return_tensors="pt").to(device)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs)
        text_embed = torch.nn.functional.normalize(text_embed, p=2, dim=-1)
    vector = text_embed.squeeze().cpu().tolist()

    top_k = 100 
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    matches = results["matches"]
    paged_matches = matches[(page-1)*page_size:page*page_size]

    return {
        "clothes": [m["id"] for m in paged_matches],
        "scores": [m["score"] for m in paged_matches],
        "total": len(matches),
        "page": page,
        "page_size": page_size
    }

@app.post("/pick")
def pick(req: PickRequest):
    try:
        make_pick(req.session_id, req.round_num, req.player, req.picks)
        return {"message": "Pick received"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank")
def rank(req: RankRequest):
    try:
        make_ranking(req.session_id, req.round_num, req.player, req.ranking)
        return {"message": "Ranking received"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
def leaderboard(session_id: str = Query(...)):
    try:
        board = get_leaderboard(session_id)
        return {"leaderboard": board}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") 

@app.get("/detailed_leaderboard")
def detailed_leaderboard(session_id: str = Query(...)):
    try:
        game_state = get_game_state(session_id)
        leaderboard = game_state['leaderboard']
        picks = game_state.get('picks', {})
        rankings = game_state.get('rankings', {})
        players = game_state['players']
        num_rounds = game_state['num_rounds']
        
        round_scores = {}
        for round_num in range(1, num_rounds + 1):
            round_scores[round_num] = {player: 0 for player in players}
            
            if round_num in rankings:
                round_rankings = rankings[round_num]
                round_picks = picks.get(round_num, {})
                
                for player, ranking in round_rankings.items():
                    n = len(ranking)
                    for rank, cloth_idx in enumerate(ranking):
                        points = n - rank

                        for other in players:
                            if other != player and other in round_picks and player in round_picks[other]:
                                if round_picks[other][player] == cloth_idx:
                                    round_scores[round_num][other] += points
        
        return {
            "leaderboard": leaderboard,
            "round_scores": round_scores,
            "players": players,
            "num_rounds": num_rounds
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/viton_preview_upload")
async def viton_preview_upload(
    cloth: str = Form(...),
    person_image: UploadFile = File(...)
):

    try:
        image_data = await person_image.read()
        person_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        cloth_path = os.path.join(CLOTHES_DIR, cloth)
        if not os.path.exists(cloth_path):
            raise HTTPException(status_code=404, detail=f"Cloth image not found: {cloth}")
        
        cloth_img = Image.open(cloth_path).convert("RGBA").resize(person_img.size)
        
        blended = Image.blend(person_img, cloth_img, alpha=0.4)
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"VITON preview upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create VITON preview: {str(e)}")

@app.get("/viton_preview")
def viton_preview(person: str, cloth: str):
    """
    Returns a simple overlay of the person and cloth images.
    """
    person_path = os.path.join(PEOPLE_DIR, person)
    cloth_path = os.path.join(CLOTHES_DIR, cloth)
    
    try:
        if not os.path.exists(person_path):
            raise HTTPException(status_code=404, detail=f"Person image not found: {person}")
        if not os.path.exists(cloth_path):
            raise HTTPException(status_code=404, detail=f"Cloth image not found: {cloth}")
        
        person_img = Image.open(person_path).convert("RGBA")
        cloth_img = Image.open(cloth_path).convert("RGBA").resize(person_img.size)
        
        blended = Image.blend(person_img, cloth_img, alpha=0.4)
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        print(f"VITON preview error: {e}")
        print(f"Person path: {person_path}")
        print(f"Cloth path: {cloth_path}")
        raise HTTPException(status_code=500, detail=f"Failed to create VITON preview: {str(e)}")

@app.get("/received_clothes")
def received_clothes(session_id: str = Query(...), round_num: int = Query(...), player: str = Query(...)):
    try:
        game_state = get_game_state(session_id)
        picks = game_state.get('picks', {}).get(round_num, {})
        players = game_state['players']
        received_clothes = []
        
        for other_player in players:
            if other_player != player and other_player in picks and player in picks[other_player]:
                cloth_idx = picks[other_player][player]
                received_clothes.append(cloth_idx)
        
        return {
            "received_clothes": received_clothes,
            "original_count": len([p for p in players if p != player]),
            "actual_count": len(received_clothes)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/generate_cloth")
async def generate_cloth(request: Request):
    import random
    try:
        body = await request.json()
        prompt = body.get('prompt', '')
    except Exception:
        prompt = ''
    clothes_dir = CLOTHES_DIR
    all_clothes = [f for f in os.listdir(clothes_dir) if f.endswith('.jpg')]
    if not all_clothes:
        raise HTTPException(status_code=404, detail="No clothes found in dataset.")
    filename = random.choice(all_clothes)
    return {"filename": filename}

@app.get("/recommend")
def recommend(top_k: int = 4):
    import os
    clothes_dir = CLOTHES_DIR
    all_clothes = [f for f in os.listdir(clothes_dir) if f.endswith('.jpg')]
    top_images = all_clothes[:top_k]
    # Return as full URLs for frontend display
    image_urls = [f"/clothes/{img}" for img in top_images]
    return {"images": image_urls}

@app.get("/api/ping")
def ping():
    return {"message": "pong"}

