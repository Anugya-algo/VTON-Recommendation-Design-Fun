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
import pinecone
from dotenv import load_dotenv
load_dotenv()
from viton_backend import run_viton_hd, VitonHDOptions
import shutil
from PIL import Image as PILImage
import io
from fastapi.responses import StreamingResponse
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set PROJECT_ROOT to the parent of VTON-Top (i.e., 12_clothes_tryon)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLOTHES_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'cloth')
PEOPLE_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'image')
CLOTH_MASK_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'cloth-mask')
OPENPOSE_JSON_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'openpose_json')
OPENPOSE_IMG_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'openpose_img')
IMAGE_PARSE_DIR = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train', 'image-parse-v3')

class CORSMiddlewareStaticFiles(StarletteStaticFiles):
    async def get_response(self, path, scope):
        print(f"[StaticFiles] get_response called for path: {path}, scope: {scope}")
        response = await super().get_response(path, scope)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        print(f"[StaticFiles] After super().get_response: path={path}, status={response.status_code}, headers={dict(response.headers)}")
        return response

app.mount("/clothes", CORSMiddlewareStaticFiles(directory=CLOTHES_DIR), name="clothes")
app.mount("/people", CORSMiddlewareStaticFiles(directory=PEOPLE_DIR), name="people")

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
    print("Received start_game request:", req)
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
        print("Error in /start_game:", e)
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
    query: str = Query("", min_length=0),
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
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    import tempfile
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        src_root = os.path.join(PROJECT_ROOT, 'clothes_tryon_dataset', 'train')
        dest_root = os.path.join(backend_dir, 'VITON-HD/datasets/single_test')
        test_dir = os.path.join(dest_root, 'test')
        os.makedirs(test_dir, exist_ok=True)
        subdirs = ['image', 'cloth', 'cloth-mask', 'openpose-json', 'openpose-img', 'image-parse']
        for sub in subdirs:
            os.makedirs(os.path.join(test_dir, sub), exist_ok=True)

        # Save uploaded person image
        person_filename = next(tempfile._get_candidate_names()) + '.jpg'
        person_path = os.path.join(test_dir, 'image', person_filename)
        with open(person_path, 'wb') as f:
            f.write(await person_image.read())

        # Save uploaded cloth image
        cloth_filename = next(tempfile._get_candidate_names()) + '.jpg'
        cloth_path = os.path.join(test_dir, 'cloth', cloth_filename)
        with open(cloth_path, 'wb') as f:
            f.write(await cloth_image.read())

        # Find corresponding cloth mask by cloth filename (must match original name in dataset)
        # For demo, try to find a mask with the same name as the uploaded cloth (if available)
        mask_src_path = os.path.join(CLOTH_MASK_DIR, cloth_image.filename)
        mask_dest_path = os.path.join(test_dir, 'cloth-mask', cloth_filename)
        if os.path.exists(mask_src_path):
            shutil.copy(mask_src_path, mask_dest_path)
        else:
            # If not found, create a blank mask
            from PIL import Image as PILImage
            PILImage.new('L', (768, 1024), color=255).save(mask_dest_path)

        # --- Run AlphaPose to generate openpose-json and openpose-img ---
        # (Assume you have a function run_alphapose that takes an image and output dirs)
        from subprocess import run
        alphapose_img_dir = os.path.join(test_dir, 'openpose-img')
        alphapose_json_dir = os.path.join(test_dir, 'openpose-json')
        os.makedirs(alphapose_img_dir, exist_ok=True)
        os.makedirs(alphapose_json_dir, exist_ok=True)
        # Call AlphaPose (update the command as needed for your setup)
        run([
            'python', '-m', 'scripts.demo_inference',
            '--cfg', 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
            '--checkpoint', 'pretrained_models/halpe26_fast_res50_256x192.pth',
            '--image', person_path,
            '--outdir', alphapose_img_dir,
            '--format', 'open',
            '--save_img'
        ], cwd=os.path.join(backend_dir, 'AlphaPose'), check=True)
        # Move JSON output to openpose-json
        for file in os.listdir(alphapose_img_dir):
            if file.endswith('_keypoints.json'):
                shutil.move(os.path.join(alphapose_img_dir, file), os.path.join(alphapose_json_dir, file))

        # --- Run image parsing model to generate image-parse ---
        # (Assume you have a function run_image_parse that takes an image and output dir)
        # For demo, just copy the person image as a placeholder
        parse_dest_path = os.path.join(test_dir, 'image-parse', person_filename.replace('.jpg', '.png'))
        PILImage.open(person_path).save(parse_dest_path)
        # TODO: Replace above with actual parsing model call

        # Write test_pairs.txt
        with open(os.path.join(dest_root, 'test_pairs.txt'), 'w') as f:
            f.write(f"{person_filename} {cloth_filename}\n")

        # Run VITON-HD
        opt = VitonHDOptions(
            name="single_test",
            dataset_dir=dest_root + '/',
            checkpoint_dir=os.path.join(backend_dir, 'VITON-HD/checkpoints/'),
            save_dir=os.path.join(backend_dir, 'VITON-HD/results/')
        )
        result_files = run_viton_hd(opt)
        if not result_files:
            raise HTTPException(status_code=500, detail="VITON-HD did not generate any result.")
        result_path = result_files[0]
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Result image not found.")
        return StreamingResponse(open(result_path, 'rb'), media_type="image/png")
    except Exception as e:
        print(f"VITON preview upload error: {e}")
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

