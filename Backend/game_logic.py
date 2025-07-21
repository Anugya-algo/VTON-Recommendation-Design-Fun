import os
import random
from PIL import Image
from typing import List, Dict, Any, Optional
import uuid

# In-memory game state storage
GAMES: Dict[str, Dict[str, Any]] = {}

CLOTHES_DIR = "../../clothes_tryon_dataset/train/cloth"


def init_game(num_players: int, num_rounds: int, timer: int, n_clothes: int = 10) -> str:
    """
    Initialize a new game session and return the session_id.
    """
    session_id = str(uuid.uuid4())
    players = [f"Player_{i+1}" for i in range(num_players)]
    all_clothes = [f for f in os.listdir(CLOTHES_DIR) if f.endswith('.jpg')]
    clothes = random.sample(all_clothes, n_clothes)
    GAMES[session_id] = {
        'players': players,
        'num_rounds': num_rounds,
        'timer': timer,
        'clothes': clothes,
        'current_round': 1,
        'picks': {},  # {round: {player: {other: cloth_idx}}}
        'rankings': {},  # {round: {player: [cloth_idx, ...]}}
        'leaderboard': {player: 0 for player in players}
    }
    return session_id


def get_clothes(session_id: str) -> List[str]:
    """
    Get the list of clothes filenames for the session.
    """
    return GAMES[session_id]['clothes']


def make_pick(session_id: str, round_num: int, player: str, picks: Dict[str, int]) -> None:
    """
    Store the picks for a player in a given round.
    picks: {other_player: cloth_idx}
    """
    game = GAMES[session_id]
    if round_num not in game['picks']:
        game['picks'][round_num] = {}
    game['picks'][round_num][player] = picks


def make_ranking(session_id: str, round_num: int, player: str, ranking: List[int]) -> None:
    """
    Store the ranking for a player in a given round.
    ranking: [cloth_idx, ...] (ordered)
    """
    game = GAMES[session_id]
    if round_num not in game['rankings']:
        game['rankings'][round_num] = {}
    
    # Get actual clothes picked for this player (including duplicates)
    picks = game['picks'].get(round_num, {})
    received_clothes = []
    
    # Get all clothes picked for this player (keep duplicates)
    for other_player in game['players']:
        if other_player != player and other_player in picks and player in picks[other_player]:
            cloth_idx = picks[other_player][player]
            received_clothes.append(cloth_idx)
    
    # Store the ranking with the actual received clothes (including duplicates)
    game['rankings'][round_num][player] = ranking
    
    # After all rankings for the round, update leaderboard
    if len(game['rankings'][round_num]) == len(game['players']):
        aggregate_scores(session_id, round_num)


def get_leaderboard(session_id: str) -> Dict[str, int]:
    """
    Get the current leaderboard for the session.
    """
    return GAMES[session_id]['leaderboard']


def aggregate_scores(session_id: str, round_num: int) -> None:
    """
    Aggregate scores for a round and update the leaderboard.
    """
    game = GAMES[session_id]
    picks = game['picks'][round_num]
    rankings = game['rankings'][round_num]
    leaderboard = game['leaderboard']
    players = game['players']
    
    for player, ranking in rankings.items():
        n = len(ranking)
        for rank, cloth_idx in enumerate(ranking):
            points = n - rank
            # Find all players who picked this cloth for the current player
            for other in players:
                if other != player and other in picks and player in picks[other]:
                    if picks[other][player] == cloth_idx:
                        leaderboard[other] += points


def get_game_state(session_id: str) -> Dict[str, Any]:
    """
    Get the full game state (for debugging or admin).
    """
    return GAMES[session_id]


def get_image_path(filename: str) -> str:
    """
    Get the absolute path to a cloth image.
    """
    return os.path.abspath(os.path.join(CLOTHES_DIR, filename))


def remove_game(session_id: str) -> None:
    """
    Remove a game session (cleanup).
    """
    if session_id in GAMES:
        del GAMES[session_id] 