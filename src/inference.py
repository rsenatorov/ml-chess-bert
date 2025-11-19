#!/usr/bin/env python3
"""
inference.py – Play vs your ChessEdgeModel with MCTS and Time Controls
- Loads the latest checkpoint from supervised training
- Enhanced Tk GUI with time controls and Monte Carlo Tree Search
- Uses the model's edge (policy), promo, and value heads

FEATURES:
- Time controls: None, Fischer (increment), Delay
- Optional MCTS tree search using model evaluation only
- Progress bar for MCTS with ETA
- Configurable game setup dialog
- Fixed promotion dialog visibility issue
"""

import os
import time
import queue
import threading
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import deque

import tkinter as tk
from tkinter import ttk, messagebox

import chess
import torch
import torch.nn.functional as F

from config import Config
from model import ChessEdgeModel
from checkpoint import CheckpointManager
from data_loader import PIECE_TOKENS, SPECIAL_FIXED, EP_FILES, HMC_BUCKETS, VOCAB


# ======================== USER-FACING SETTINGS ========================
class InferenceConfig:
    # Paths
    run_dir: str = "runs/chess_finetune"          # root of training run (expects ./checkpoints inside)
    checkpoint: Optional[str] = None     # optional explicit path to a checkpoint .pt file

    # AI behavior
    use_top1: bool = True                # True = deterministic top-1 legal move
    temperature: float = 0.2             # used only if use_top1=False
    ai_think_time: float = 0.4           # UX: minimum "thinking" time before AI moves

    # Side to play for human
    player_color: str = "white"          # "white", "black", or "random"

    # Board look
    light_color: str = "#F0D9B5"
    dark_color: str = "#B58863"
    highlight_color: str = "#646f40"
    last_move_color: str = "#ccc23b"
    square_size: int = 80
    piece_font_family: str = "Arial"
    piece_font_size: int = 50


# ======================== PIECE GLYPHS ========================
PIECE_UNICODE = {
    chess.KING:   {"white": "♔", "black": "♚"},
    chess.QUEEN:  {"white": "♕", "black": "♛"},
    chess.ROOK:   {"white": "♖", "black": "♜"},
    chess.BISHOP: {"white": "♗", "black": "♝"},
    chess.KNIGHT: {"white": "♘", "black": "♞"},
    chess.PAWN:   {"white": "♙", "black": "♟"},
}


# ======================== CHESS CLOCK ========================
class ChessClock:
    """Manages time control for both players with increment/delay support"""
    
    def __init__(self, initial_ms: int = 600000, increment_ms: int = 0, 
                 delay_ms: int = 0, mode: str = 'fischer'):
        """
        Args:
            initial_ms: Starting time in milliseconds (default 10 minutes)
            increment_ms: Time added after each move (Fischer)
            delay_ms: Delay before time starts counting (Delay mode)
            mode: 'none', 'fischer', or 'delay'
        """
        self.initial_ms = initial_ms
        self.increment_ms = increment_ms
        self.delay_ms = delay_ms
        self.mode = mode
        
        self.white_ms = initial_ms
        self.black_ms = initial_ms
        self.active_color: Optional[chess.Color] = None
        self.turn_start_time: Optional[float] = None
        self.paused = False
        
    def start_turn(self, color: chess.Color):
        """Start clock for given color"""
        if self.mode == 'none':
            return
        self.active_color = color
        self.turn_start_time = time.time()
        self.paused = False
        
    def stop_turn(self, color: chess.Color) -> int:
        """Stop clock and return time used in ms"""
        if self.mode == 'none' or self.turn_start_time is None:
            return 0
            
        elapsed_ms = int((time.time() - self.turn_start_time) * 1000)
        
        # Apply delay (time doesn't count until delay exhausted)
        if self.mode == 'delay' and elapsed_ms <= self.delay_ms:
            elapsed_ms = 0
        elif self.mode == 'delay':
            elapsed_ms -= self.delay_ms
            
        # Deduct time
        if color == chess.WHITE:
            self.white_ms -= elapsed_ms
        else:
            self.black_ms -= elapsed_ms
            
        # Add increment (Fischer)
        if self.mode == 'fischer':
            if color == chess.WHITE:
                self.white_ms += self.increment_ms
            else:
                self.black_ms += self.increment_ms
                
        self.active_color = None
        self.turn_start_time = None
        return elapsed_ms
        
    def pause(self):
        """Pause the active clock"""
        self.paused = True
        
    def resume(self):
        """Resume the active clock"""
        if self.active_color is not None:
            self.turn_start_time = time.time()
            self.paused = False
        
    def get_display_time(self, color: chess.Color) -> str:
        """Get display string MM:SS or HH:MM:SS"""
        ms = self.white_ms if color == chess.WHITE else self.black_ms
        
        # If active, calculate current time
        if self.active_color == color and self.turn_start_time and not self.paused:
            elapsed_ms = int((time.time() - self.turn_start_time) * 1000)
            if self.mode == 'delay':
                # During delay, don't deduct
                if elapsed_ms > self.delay_ms:
                    ms -= (elapsed_ms - self.delay_ms)
            else:
                ms -= elapsed_ms
                
        ms = max(0, ms)
        total_sec = ms // 1000
        
        hours = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        
        if hours > 0:
            return f"{hours}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"
        
    def is_flagged(self, color: chess.Color) -> bool:
        """Check if time expired"""
        if self.mode == 'none':
            return False
            
        ms = self.white_ms if color == chess.WHITE else self.black_ms
        
        if self.active_color == color and self.turn_start_time and not self.paused:
            elapsed_ms = int((time.time() - self.turn_start_time) * 1000)
            if self.mode == 'delay':
                if elapsed_ms > self.delay_ms:
                    ms -= (elapsed_ms - self.delay_ms)
            else:
                ms -= elapsed_ms
                
        return ms <= 0


# ======================== BOARD ENCODING ========================
def encode_board_to_tokens(board: chess.Board) -> List[int]:
    """
    Encode a chess.Board into the 72-token format expected by the model.
    Matches the encoding used in training data.
    """
    def piece_token(piece: Optional[chess.Piece]) -> str:
        if piece is None:
            return "EMPTY"
        return f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"
    
    tokens = ["CLS"]
    
    # 64 squares: a1..h8 (rank 1..8, file a..h within each rank)
    for rank in range(1, 9):
        for file_idx in range(8):
            sq = chess.square(file_idx, rank - 1)
            tokens.append(piece_token(board.piece_at(sq)))
    
    # Turn
    tokens.append("TURN_W" if board.turn == chess.WHITE else "TURN_B")
    
    # Castling rights (4 flags)
    tokens.append("WK_CASTLE_YES" if board.has_kingside_castling_rights(chess.WHITE) else "WK_CASTLE_NO")
    tokens.append("WQ_CASTLE_YES" if board.has_queenside_castling_rights(chess.WHITE) else "WQ_CASTLE_NO")
    tokens.append("BK_CASTLE_YES" if board.has_kingside_castling_rights(chess.BLACK) else "BK_CASTLE_NO")
    tokens.append("BQ_CASTLE_YES" if board.has_queenside_castling_rights(chess.BLACK) else "BQ_CASTLE_NO")
    
    # En passant
    if board.ep_square is None:
        tokens.append("EP_NONE")
    else:
        file_idx = chess.square_file(board.ep_square)
        tokens.append(f"EP_{'ABCDEFGH'[file_idx]}")
    
    # Halfmove clock (bucketed)
    hmc = board.halfmove_clock
    bucket_idx = min(hmc // 10, 9)
    tokens.append(f"HMC_{bucket_idx*10:02d}_{bucket_idx*10+9:02d}")
    
    assert len(tokens) == 72, f"Expected 72 tokens, got {len(tokens)}"
    
    # Convert to token IDs
    return [VOCAB[t] for t in tokens]


# ======================== MCTS IMPLEMENTATION ========================
class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    
    def __init__(self, board: chess.Board, move: Optional[chess.Move] = None, parent=None):
        self.board = board.copy() if board else None
        self.move = move  # Move that led to this position
        self.parent = parent
        self.children: List[MCTSNode] = []
        
        self.visit_count = 0
        self.value_sum = 0.0  # Sum of values from white's perspective
        self.prior = 0.0  # Policy probability from model
        self.is_expanded = False
        
    def value(self) -> float:
        """Average value from white's perspective"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def uct_score(self, parent_visits: int, parent_turn: chess.Color, c_puct: float = 1.4) -> float:
        """
        UCT score for selection
        Args:
            parent_visits: Number of visits to parent node
            parent_turn: Whose turn it is at the parent (who is choosing this child)
            c_puct: Exploration constant
        """
        if self.visit_count == 0:
            return float('inf')
            
        # Value is always from white's perspective
        white_pov_value = self.value()
        
        # If Black is choosing (parent_turn == BLACK), they want LOW white POV value
        # So we negate to make it maximization
        if parent_turn == chess.BLACK:
            exploitation = -white_pov_value
        else:
            exploitation = white_pov_value
        
        # UCT exploration term
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return exploitation + exploration


class MCTSSearcher:
    """Monte Carlo Tree Search using model evaluation only (no minimax)"""
    
    def __init__(self, inference, num_simulations: int = 100, c_puct: float = 1.4):
        self.inference = inference
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.nodes_searched = 0
        self.cancelled = False
        
    def search(self, board: chess.Board, progress_callback=None, 
               time_limit_ms: Optional[int] = None) -> Tuple[chess.Move, Dict]:
        """
        Run MCTS and return best move with statistics
        
        Args:
            board: Current board position
            progress_callback: Function(current, total, eta) for progress updates
            time_limit_ms: Maximum time in milliseconds
            
        Returns:
            (best_move, statistics_dict)
        """
        root = MCTSNode(board)
        start_time = time.time()
        self.nodes_searched = 0
        self.cancelled = False
        
        for sim in range(self.num_simulations):
            if self.cancelled:
                break
                
            # Check time limit
            if time_limit_ms:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > time_limit_ms:
                    break
                
            # Single MCTS iteration: Select -> Expand -> Evaluate -> Backpropagate
            node = self._select(root)
            value = self._expand_and_evaluate(node)
            self._backpropagate(node, value)
            
            self.nodes_searched += 1
            
            # Progress callback
            if progress_callback and sim % 5 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / max(1, sim)) * (self.num_simulations - sim) if sim > 0 else 0
                progress_callback(sim + 1, self.num_simulations, eta)
                
        # Select best move (most visited child)
        if not root.children:
            # Fallback: no expansion happened
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available")
            return legal_moves[0], {'nodes_searched': 0}
            
        best_child = max(root.children, key=lambda c: c.visit_count)
        
        stats = {
            'nodes_searched': self.nodes_searched,
            'best_move_visits': best_child.visit_count,
            'best_move_value': best_child.value(),
            'total_time': time.time() - start_time,
            'pv': self._get_principal_variation(root, depth=5)
        }
        
        return best_child.move, stats
        
    def cancel(self):
        """Cancel the current search"""
        self.cancelled = True
        
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select leaf node using UCT"""
        while node.is_expanded and node.children:
            # If game over, return this node
            if node.board.is_game_over():
                return node
            # Select best child by UCT
            node = max(node.children, 
                      key=lambda c: c.uct_score(node.visit_count, node.board.turn, self.c_puct))
        return node
        
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Expand node and evaluate using model (returns value from white's POV)"""
        board = node.board
        
        # Terminal node evaluation
        if board.is_checkmate():
            # board.turn is the side that got checkmated
            # If White got checkmated, return -1.0 (bad for White)
            # If Black got checkmated, return 1.0 (good for White)
            return -1.0 if board.turn == chess.WHITE else 1.0
        elif board.is_game_over():  # Stalemate, insufficient material, etc.
            return 0.0
            
        # Get model evaluation
        tokens = torch.tensor([encode_board_to_tokens(board)], 
                             dtype=torch.long, device=self.inference.device)
        
        with torch.no_grad():
            edge_logits, promo_logits, value_logits, _ = self.inference.model(tokens)
            
        # Value from white's perspective
        value_probs = torch.softmax(value_logits[0], dim=0)
        # probs[0] = P(Black wins), probs[1] = P(Draw), probs[2] = P(White wins)
        value = float(value_probs[2].item() - value_probs[0].item())
        
        # Expand node with policy priors
        if not node.is_expanded:
            self._expand_node(node, edge_logits[0], promo_logits[0])
            
        return value
        
    def _expand_node(self, node: MCTSNode, edge_logits: torch.Tensor, promo_logits: torch.Tensor):
        """Expand node with children using policy priors"""
        board = node.board
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            node.is_expanded = True
            return
            
        # Get policy probabilities for legal moves
        legal_mask = torch.zeros(4096, dtype=torch.bool, device=edge_logits.device)
        move_to_idx: Dict[chess.Move, int] = {}
        edge_to_moves: Dict[int, List[chess.Move]] = {}
        
        for mv in legal_moves:
            idx = mv.from_square * 64 + mv.to_square
            legal_mask[idx] = True
            move_to_idx[mv] = idx
            edge_to_moves.setdefault(idx, []).append(mv)
            
        masked_logits = edge_logits.masked_fill(~legal_mask, float('-inf'))
        policy_probs = torch.softmax(masked_logits, dim=0)
        
        # Handle promotions: for each edge with multiple moves, distribute probability
        final_probs: Dict[chess.Move, float] = {}
        
        for edge_idx, moves in edge_to_moves.items():
            edge_prob = float(policy_probs[edge_idx].item())
            
            if len(moves) == 1:
                final_probs[moves[0]] = edge_prob
            else:
                # Multiple promotion moves on this edge
                # Use promo head to distribute probability
                promo_scores = promo_logits[edge_idx]  # [4]
                promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
                
                # Mask to legal promotions
                promo_mask = torch.zeros(4, dtype=torch.bool, device=promo_scores.device)
                for mv in moves:
                    if mv.promotion in promo_map:
                        promo_mask[promo_map[mv.promotion]] = True
                        
                masked_promo = promo_scores.masked_fill(~promo_mask, float('-inf'))
                promo_probs = torch.softmax(masked_promo, dim=0)
                
                for mv in moves:
                    if mv.promotion in promo_map:
                        promo_prob = float(promo_probs[promo_map[mv.promotion]].item())
                        final_probs[mv] = edge_prob * promo_prob
                    else:
                        final_probs[mv] = edge_prob / len(moves)  # Fallback
        
        # Create child nodes with priors
        for mv in legal_moves:
            child_board = board.copy()
            child_board.push(mv)
            child = MCTSNode(child_board, mv, parent=node)
            child.prior = final_probs.get(mv, 1e-8)
            node.children.append(child)
            
        node.is_expanded = True
        
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value (white's POV) up the tree"""
        while node is not None:
            node.visit_count += 1
            # Value is always from white's perspective
            node.value_sum += value
            node = node.parent
            
    def _get_principal_variation(self, node: MCTSNode, depth: int) -> List[str]:
        """Get principal variation (most visited path)"""
        pv = []
        for _ in range(depth):
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.visit_count)
            if node.move:
                # Get SAN notation
                parent_board = node.parent.board if node.parent else chess.Board()
                pv.append(parent_board.san(node.move))
        return pv


# ======================== INFERENCE CORE ========================
class ChessInference:
    """Wraps model loading + move/eval prediction."""

    def __init__(self, run_dir: str, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.cfg = Config()  # Load training config for model architecture
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build & eval-mode the model
        self.model = ChessEdgeModel(self.cfg).to(self.device)
        self.model.eval()

        # Resolve checkpoint to load
        ckpt_to_load = self._resolve_checkpoint_path(run_dir, checkpoint_path)
        print(f"[inference] Loading checkpoint: {ckpt_to_load}")
        payload = torch.load(ckpt_to_load, map_location=self.device)

        # Get the state dict
        state_dict = self._extract_state_dict(payload)
        state_dict = self._strip_module_prefix(state_dict)
        self.model.load_state_dict(state_dict, strict=True)

        # Print checkpoint info
        epoch = payload.get("state", {}).get("epoch")
        if epoch is not None:
            print(f"[inference] Loaded checkpoint from epoch {epoch}")
        else:
            print("[inference] Loaded checkpoint (epoch unknown)")

    # ---------- checkpoint helpers ----------
    def _resolve_checkpoint_path(self, run_dir: str, explicit: Optional[str]) -> str:
        """
        Priority:
          1) explicit arg
          2) runs/chess/checkpoints/checkpoint_latest.pt
          3) newest runs/chess/checkpoints/checkpoint_epoch_*.pt by epoch number
        """
        if explicit:
            p = Path(explicit)
            if p.exists():
                return str(p.resolve())
            raise FileNotFoundError(f"Explicit checkpoint not found: {explicit}")

        ckpt_dir = Path(run_dir) / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        latest = ckpt_dir / "checkpoint_epoch_9.pt"
        if latest.exists():
            return str(latest.resolve())

        # Find numbered checkpoints
        numbered = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if numbered:
            def parse_epoch(path: Path) -> int:
                try:
                    return int(path.stem.split("_")[-1])
                except Exception:
                    return -1
            best = max(numbered, key=parse_epoch)
            return str(best.resolve())

        raise FileNotFoundError(
            f"No checkpoint found in: {ckpt_dir}\n"
            f"Expected checkpoint_latest.pt or checkpoint_epoch_*.pt"
        )

    @staticmethod
    def _extract_state_dict(payload: Dict) -> Dict:
        """Try common keys used when saving models."""
        if isinstance(payload, dict):
            for key in ("model", "state_dict", "model_state", "model_state_dict"):
                if key in payload and isinstance(payload[key], dict):
                    return payload[key]
        # If state dict was saved directly
        if isinstance(payload, dict):
            some_tensor_vals = [isinstance(v, torch.Tensor) for v in list(payload.values())[:10]]
            if some_tensor_vals and all(some_tensor_vals):
                return payload
        raise RuntimeError("Could not find model state_dict in checkpoint payload.")

    @staticmethod
    def _strip_module_prefix(state_dict: Dict) -> Dict:
        """Remove 'module.' prefix (from DDP) if present."""
        if not state_dict:
            return state_dict
        if not any(k.startswith("module.") for k in state_dict.keys()):
            return state_dict
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate_position(self, board: chess.Board) -> Tuple[float, float, float, float]:
        """
        Return model eval from White's perspective with probabilities.
        
        Returns:
            score: Continuous score in [-1, 1] from White's POV
            p_white_win: Probability White wins [0, 1]
            p_draw: Probability of draw [0, 1]
            p_black_win: Probability Black wins [0, 1]
        """
        tokens = torch.tensor([encode_board_to_tokens(board)], dtype=torch.long, device=self.device)
        _, _, value_logits, _ = self.model(tokens)
        probs = torch.softmax(value_logits[0], dim=0)
        
        p_black_win = float(probs[0].item())
        p_draw = float(probs[1].item())
        p_white_win = float(probs[2].item())
        
        # Continuous score: P(white win) - P(black win)
        score = p_white_win - p_black_win
        
        return score, p_white_win, p_draw, p_black_win

    # ---------- move prediction ----------
    @torch.no_grad()
    def predict_move(
        self,
        board: chess.Board,
        use_top1: bool = True,
        temperature: float = 0.2,
    ) -> Tuple[chess.Move, float, float]:
        """
        Choose a legal move using policy head (and promo head if needed).
        Returns: (selected_move, edge_probability, value_score)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available.")

        tokens = torch.tensor([encode_board_to_tokens(board)], dtype=torch.long, device=self.device)
        edge_logits, promo_logits, value_logits, _ = self.model(tokens)
        edge_logits = edge_logits[0]        # [4096]
        promo_logits = promo_logits[0]      # [4096, 4]

        # Build mask over edges for current legal set
        legal_mask = torch.zeros(4096, dtype=torch.bool, device=self.device)
        edge_to_moves: Dict[int, List[chess.Move]] = {}
        for mv in legal_moves:
            idx = mv.from_square * 64 + mv.to_square
            legal_mask[idx] = True
            edge_to_moves.setdefault(idx, []).append(mv)

        masked_logits = edge_logits.masked_fill(~legal_mask, float("-inf"))

        # Pick an edge
        if use_top1:
            best_edge_idx = int(masked_logits.argmax().item())
            edge_probs = F.softmax(masked_logits, dim=0)
        else:
            edge_probs = F.softmax(masked_logits / max(1e-6, float(temperature)), dim=0)
            best_edge_idx = int(torch.multinomial(edge_probs, 1).item())

        # Decide exact move (handle promotions if multiple moves on this edge)
        candidates = edge_to_moves.get(best_edge_idx, [])
        if not candidates:
            mv = legal_moves[0]
        elif len(candidates) == 1:
            mv = candidates[0]
        else:
            # Promotion case: pick best legal promotion with promo head
            promo_map = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
            mask = torch.zeros(4, dtype=torch.bool, device=self.device)
            for m in candidates:
                if m.promotion in promo_map:
                    mask[promo_map[m.promotion]] = True
            promo_scores = promo_logits[best_edge_idx].masked_fill(~mask, float("-inf"))
            best_promo_idx = int(promo_scores.argmax().item())
            idx_to_piece = {0: chess.QUEEN, 1: chess.ROOK, 2: chess.BISHOP, 3: chess.KNIGHT}
            want = idx_to_piece[best_promo_idx]
            mv = next((m for m in candidates if m.promotion == want), candidates[0])

        edge_conf = float(edge_probs[best_edge_idx].item())
        value_score, _, _, _ = self.evaluate_position(board)
        return mv, edge_conf, value_score


# ======================== GAME SETUP DIALOG ========================
class GameSetupDialog:
    """Modal dialog for pre-game configuration"""
    
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("New Game Setup")
        self.dialog.geometry("520x780")
        self.dialog.configure(bg="#2b2b2b")
        self.dialog.transient(parent)
        
        self._create_widgets()
        self._center_window()
        
    def _create_widgets(self):
        # Title
        tk.Label(self.dialog, text="Chess Game Setup", font=("Arial", 18, "bold"),
                bg="#2b2b2b", fg="white").pack(pady=20)
        
        # Color selection
        color_frame = tk.LabelFrame(self.dialog, text="Your Color", font=("Arial", 12, "bold"),
                                   bg="#2b2b2b", fg="white", padx=20, pady=10)
        color_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.color_var = tk.StringVar(value="white")
        tk.Radiobutton(color_frame, text="♔ White", variable=self.color_var, value="white",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a",
                      font=("Arial", 11)).pack(anchor=tk.W)
        tk.Radiobutton(color_frame, text="♚ Black", variable=self.color_var, value="black",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a",
                      font=("Arial", 11)).pack(anchor=tk.W)
        tk.Radiobutton(color_frame, text="🎲 Random", variable=self.color_var, value="random",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a",
                      font=("Arial", 11)).pack(anchor=tk.W)
        
        # Time control
        time_frame = tk.LabelFrame(self.dialog, text="Time Control", font=("Arial", 12, "bold"),
                                  bg="#2b2b2b", fg="white", padx=20, pady=10)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Time mode
        self.time_mode = tk.StringVar(value="fischer")
        tk.Radiobutton(time_frame, text="No Time Limit", variable=self.time_mode, value="none",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a").pack(anchor=tk.W)
        tk.Radiobutton(time_frame, text="Fischer (Increment)", variable=self.time_mode, value="fischer",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a").pack(anchor=tk.W)
        tk.Radiobutton(time_frame, text="Delay", variable=self.time_mode, value="delay",
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a").pack(anchor=tk.W)
        
        # Starting time
        tk.Label(time_frame, text="Starting Time (minutes):", bg="#2b2b2b", fg="white").pack(anchor=tk.W, pady=(10,0))
        self.time_scale = tk.Scale(time_frame, from_=1, to=60, orient=tk.HORIZONTAL,
                                   bg="#2b2b2b", fg="white", highlightthickness=0, length=350)
        self.time_scale.set(10)
        self.time_scale.pack(fill=tk.X)
        
        # Increment
        tk.Label(time_frame, text="Increment (seconds):", bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.increment_scale = tk.Scale(time_frame, from_=0, to=60, orient=tk.HORIZONTAL,
                                       bg="#2b2b2b", fg="white", highlightthickness=0, length=350)
        self.increment_scale.set(0)
        self.increment_scale.pack(fill=tk.X)
        
        # Delay
        tk.Label(time_frame, text="Delay (seconds):", bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.delay_scale = tk.Scale(time_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                    bg="#2b2b2b", fg="white", highlightthickness=0, length=350)
        self.delay_scale.set(0)
        self.delay_scale.pack(fill=tk.X)
        
        
        # Position Setup
        position_frame = tk.LabelFrame(self.dialog, text="Starting Position", font=("Arial", 12, "bold"),
                                      bg="#2b2b2b", fg="white", padx=20, pady=10)
        position_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(position_frame, text="FEN (leave empty for standard start):", 
                bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.fen_entry = tk.Entry(position_frame, width=50, font=("Arial", 10),
                                 bg="#1a1a1a", fg="white", insertbackground="white")
        self.fen_entry.pack(fill=tk.X, pady=5)
        
        # AI Settings
        ai_frame = tk.LabelFrame(self.dialog, text="AI Settings", font=("Arial", 12, "bold"),
                                bg="#2b2b2b", fg="white", padx=20, pady=10)
        ai_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.use_mcts = tk.BooleanVar(value=False)
        tk.Checkbutton(ai_frame, text="Use Tree Search (MCTS)", variable=self.use_mcts,
                      bg="#2b2b2b", fg="white", selectcolor="#4a4a4a",
                      font=("Arial", 11), command=self._toggle_mcts).pack(anchor=tk.W)
        
        self.mcts_frame = tk.Frame(ai_frame, bg="#2b2b2b")
        
        tk.Label(self.mcts_frame, text="Simulations per move:", bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.sims_scale = tk.Scale(self.mcts_frame, from_=50, to=1000, orient=tk.HORIZONTAL,
                                   bg="#2b2b2b", fg="white", highlightthickness=0, length=350)
        self.sims_scale.set(100)
        self.sims_scale.pack(fill=tk.X)
        # Hidden by default
        
        # Buttons
        btn_frame = tk.Frame(self.dialog, bg="#2b2b2b")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Start Game", command=self._on_start,
                 bg="#4a90e2", fg="white", font=("Arial", 12, "bold"),
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self._on_cancel,
                 bg="#e24a4a", fg="white", font=("Arial", 12, "bold"),
                 width=12, height=2).pack(side=tk.LEFT, padx=5)
        
    def _toggle_mcts(self):
        if self.use_mcts.get():
            self.mcts_frame.pack(fill=tk.X, pady=5)
        else:
            self.mcts_frame.pack_forget()
            
    def _center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
    def _on_start(self):
        # Validate FEN if provided
        fen_string = self.fen_entry.get().strip()
        if fen_string:
            try:
                # Try to create a board from the FEN to validate it
                chess.Board(fen_string)
            except ValueError:
                messagebox.showerror("Invalid FEN", 
                                   "The FEN string you entered is invalid. Please check and try again.")
                return
        
        self.result = {
            'color': self.color_var.get(),
            'time_mode': self.time_mode.get(),
            'initial_time_min': self.time_scale.get(),
            'increment_sec': self.increment_scale.get(),
            'delay_sec': self.delay_scale.get(),
            'use_mcts': self.use_mcts.get(),
            'mcts_sims': self.sims_scale.get() if self.use_mcts.get() else 0,
            'fen': fen_string if fen_string else None
        }
        self.dialog.destroy()
        
    def _on_cancel(self):
        self.result = None
        self.dialog.destroy()
        
    def show(self):
        """Show dialog and return configuration dict or None"""
        self.dialog.wait_window()
        return self.result


# ======================== TK GUI ========================
class ChessGUI:
    def __init__(self, inference: ChessInference, config: InferenceConfig):
        self.inference = inference
        self.config = config

        self.board = chess.Board()
        self.player_is_white = True
        self.is_player_turn = True
        self.game_over = False
        self.last_move: Optional[chess.Move] = None

        self.selected_square: Optional[int] = None
        self.legal_from_selected: List[chess.Move] = []

        self.move_queue: "queue.Queue[bool]" = queue.Queue()

        self.piece_font = (self.config.piece_font_family, self.config.piece_font_size)
        self.dot_font = (self.config.piece_font_family, max(12, self.config.piece_font_size // 2))

        # New: Game configuration
        self.clock: Optional[ChessClock] = None
        self.use_mcts = False
        self.mcts_simulations = 100
        self.mcts_searcher: Optional[MCTSSearcher] = None
        
        # Timer update job
        self.timer_job: Optional[str] = None
        
        # MCTS progress widgets
        self.mcts_progress_frame: Optional[tk.Frame] = None
        self.mcts_progress_bar: Optional[ttk.Progressbar] = None
        self.mcts_status_label: Optional[tk.Label] = None

        self._init_window()
        self._init_menu()
        self._init_board()
        self._create_mcts_progress_widget()
        self._refresh()

        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        self.ai_thread.start()

    # ---------- UI setup ----------
    def _init_window(self):
        self.root = tk.Tk()
        self.root.title("Chess AI – Enhanced")
        self.root.resizable(False, False)

        board_px = self.config.square_size * 8
        self.root.geometry(f"{board_px + 280}x{board_px + 100}")

        self.main_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: board + eval bar
        left = tk.Frame(self.main_frame, bg="#2b2b2b")
        left.pack(side=tk.LEFT, padx=20, pady=20)

        self.board_frame = tk.Frame(left, highlightthickness=2, highlightbackground="#8B4513")
        self.board_frame.pack(side=tk.LEFT)

        eval_frame = tk.Frame(left, bg="#2b2b2b")
        eval_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(eval_frame, text="EVALUATION", font=("Arial", 9, "bold"), bg="#2b2b2b", fg="white").pack()

        score_wrap = tk.Frame(eval_frame, bg="#1a1a1a", highlightthickness=1, highlightbackground="#555555")
        score_wrap.pack(pady=5)
        self.eval_text = tk.Label(score_wrap, text="+0.00", font=("Arial", 20, "bold"), bg="#1a1a1a", fg="white", width=6)
        self.eval_text.pack(padx=5, pady=5)
        
        self.prob_text = tk.Label(eval_frame, text="W:0% D:0% B:0%", font=("Arial", 8), bg="#2b2b2b", fg="white")
        self.prob_text.pack()

        tk.Label(eval_frame, text="♔ White", font=("Arial", 10, "bold"), bg="#2b2b2b", fg="#4a90e2").pack()
        tk.Label(eval_frame, text="+1.0", font=("Arial", 8), bg="#2b2b2b", fg="white").pack()

        self.eval_canvas = tk.Canvas(eval_frame, width=30, height=self.config.square_size * 8,
                                     bg="#1a1a1a", highlightthickness=1, highlightbackground="#555555")
        self.eval_canvas.pack(pady=5)

        H = self.config.square_size * 8
        for i in range(H):
            ratio = i / H
            if ratio < 0.5:
                r = int(70 + (136 - 70) * (ratio * 2))
                g = int(144 + (136 - 144) * (ratio * 2))
                b = int(226 + (136 - 226) * (ratio * 2))
            else:
                r = int(136 + (226 - 136) * ((ratio - 0.5) * 2))
                g = int(136 + (70 - 136) * ((ratio - 0.5) * 2))
                b = int(136 + (70 - 136) * ((ratio - 0.5) * 2))
            self.eval_canvas.create_line(0, i, 30, i, fill=f"#{r:02x}{g:02x}{b:02x}")
        self.eval_canvas.create_line(0, H // 2, 30, H // 2, fill="white", width=3)
        self.eval_indicator = self.eval_canvas.create_line(0, H // 2, 30, H // 2, fill="yellow", width=5)

        tk.Label(eval_frame, text="-1.0", font=("Arial", 8), bg="#2b2b2b", fg="white").pack()
        tk.Label(eval_frame, text="♚ Black", font=("Arial", 10, "bold"), bg="#2b2b2b", fg="#e24a4a").pack()

        # Right: info panel
        info = tk.Frame(self.main_frame, bg="#2b2b2b", width=170)
        info.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=20)
        info.pack_propagate(False)

        self.status = tk.Label(info, text="Your turn", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white")
        self.status.pack(pady=10)
        
        # Timer displays
        timer_container = tk.Frame(info, bg="#2b2b2b")
        timer_container.pack(fill=tk.X, pady=10)
        
        black_timer_frame = tk.Frame(timer_container, bg="#1a1a1a", highlightthickness=1, highlightbackground="#555")
        black_timer_frame.pack(fill=tk.X, pady=2)
        tk.Label(black_timer_frame, text="♚", font=("Arial", 12, "bold"),
                bg="#1a1a1a", fg="white", width=2).pack(side=tk.LEFT, padx=5)
        self.black_timer_label = tk.Label(black_timer_frame, text="10:00", 
                                          font=("Courier", 16, "bold"),
                                          bg="#1a1a1a", fg="white")
        self.black_timer_label.pack(side=tk.RIGHT, padx=5)
        
        white_timer_frame = tk.Frame(timer_container, bg="#1a1a1a", highlightthickness=1, highlightbackground="#555")
        white_timer_frame.pack(fill=tk.X, pady=2)
        tk.Label(white_timer_frame, text="♔", font=("Arial", 12, "bold"),
                bg="#1a1a1a", fg="white", width=2).pack(side=tk.LEFT, padx=5)
        self.white_timer_label = tk.Label(white_timer_frame, text="10:00",
                                          font=("Courier", 16, "bold"),
                                          bg="#1a1a1a", fg="white")
        self.white_timer_label.pack(side=tk.RIGHT, padx=5)
        
        tk.Label(info, text="Position (FEN)", font=("Arial", 10, "bold"), bg="#2b2b2b", fg="white").pack(pady=(10, 2))
        fen_frame = tk.Frame(info, bg="#2b2b2b")
        fen_frame.pack(fill=tk.X, padx=5)
        
        self.fen_text = tk.Text(fen_frame, height=3, width=20, wrap=tk.WORD, font=("Courier", 8))
        self.fen_text.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(fen_frame, text="Copy FEN", command=self._copy_fen, 
                 bg="#4a4a4a", fg="white", font=("Arial", 8)).pack(side=tk.TOP, pady=2, fill=tk.X)

        tk.Label(info, text="Moves", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="white").pack(pady=(10, 5))
        history_frame = tk.Frame(info, bg="white", highlightthickness=1)
        history_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history = tk.Text(history_frame, width=16, height=10, yscrollcommand=scrollbar.set, font=("Courier", 10))
        self.history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history.yview)
        
        tk.Button(info, text="Copy Moves", command=self._copy_moves, 
                 bg="#4a4a4a", fg="white", font=("Arial", 9)).pack(pady=2, fill=tk.X)

        tk.Button(info, text="New Game", command=self._new_game, bg="#4a4a4a", fg="white",
                  font=("Arial", 11, "bold")).pack(pady=5, fill=tk.X)
        tk.Button(info, text="Undo Move", command=self._undo_pair, bg="#4a4a4a", fg="white",
                  font=("Arial", 11, "bold")).pack(pady=2, fill=tk.X)

    def _init_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self._new_game)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)

    def _init_board(self):
        self.squares: Dict[int, tk.Frame] = {}
        self.square_labels: Dict[int, tk.Label] = {}

        for row in range(8):
            for col in range(8):
                sq = chess.square(col, 7 - row)
                is_light = (row + col) % 2 == 0
                base = self.config.light_color if is_light else self.config.dark_color

                cell = tk.Frame(self.board_frame, width=self.config.square_size, height=self.config.square_size, bg=base)
                cell.grid(row=row, column=col)
                cell.grid_propagate(False)

                label = tk.Label(cell, text="", font=self.piece_font, bg=base, fg="black")
                label.place(relx=0.5, rely=0.5, anchor="center")

                self.squares[sq] = cell
                self.square_labels[sq] = label

                cell.bind("<Button-1>", lambda e, s=sq: self._click_square(s))
                label.bind("<Button-1>", lambda e, s=sq: self._click_square(s))

    def _create_mcts_progress_widget(self):
        """Create MCTS progress display (hidden by default)"""
        self.mcts_progress_frame = tk.Frame(self.main_frame, bg="#2b2b2b",
                                           highlightthickness=3, highlightbackground="#4a90e2")
        self.mcts_progress_frame.place_forget()  # Hidden by default
        
        tk.Label(self.mcts_progress_frame, text="AI Analyzing with Tree Search...", 
                font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=15, padx=20)
        
        self.mcts_progress_bar = ttk.Progressbar(self.mcts_progress_frame, 
                                                 length=350, mode='determinate')
        self.mcts_progress_bar.pack(padx=20, pady=10)
        
        self.mcts_status_label = tk.Label(self.mcts_progress_frame, 
                                          text="0/100 nodes | ETA: --",
                                          font=("Arial", 11), bg="#2b2b2b", fg="white")
        self.mcts_status_label.pack(pady=10)
        
        tk.Button(self.mcts_progress_frame, text="Cancel Search", 
                 command=self._cancel_mcts,
                 bg="#e24a4a", fg="white", font=("Arial", 10, "bold"),
                 width=15).pack(pady=15)

    def _show_mcts_progress(self):
        """Show MCTS progress overlay"""
        if self.mcts_progress_frame:
            # Center on main frame
            self.mcts_progress_frame.place(relx=0.5, rely=0.5, anchor="center")
            self.mcts_progress_bar['value'] = 0
            self.mcts_status_label.config(text="Starting search...")
            
    def _hide_mcts_progress(self):
        """Hide MCTS progress overlay"""
        if self.mcts_progress_frame:
            self.mcts_progress_frame.place_forget()
            
    def _update_mcts_progress(self, current: int, total: int, eta: float):
        """Update MCTS progress display"""
        if self.mcts_progress_bar and self.mcts_status_label:
            progress = (current / total) * 100
            self.mcts_progress_bar['value'] = progress
            self.mcts_status_label.config(text=f"{current}/{total} nodes | ETA: {eta:.1f}s")
            
    def _cancel_mcts(self):
        """Cancel ongoing MCTS search"""
        if self.mcts_searcher:
            self.mcts_searcher.cancel()
        self._hide_mcts_progress()

    # ---------- board helpers ----------
    @staticmethod
    def _base_color(cfg: InferenceConfig, sq: int) -> str:
        row, col = 7 - chess.square_rank(sq), chess.square_file(sq)
        return cfg.light_color if (row + col) % 2 == 0 else cfg.dark_color

    def _update_fen_display(self):
        fen = self.board.fen()
        self.fen_text.delete(1.0, tk.END)
        self.fen_text.insert(1.0, fen)

    def _copy_fen(self):
        fen = self.board.fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(fen)
        self.status.config(text="FEN copied!")
        self.root.after(2000, lambda: self.status.config(text="Your turn" if self.is_player_turn else "AI thinking..."))

    def _copy_moves(self):
        moves = self.history.get(1.0, tk.END).strip()
        if moves:
            self.root.clipboard_clear()
            self.root.clipboard_append(moves)
            self.status.config(text="Moves copied!")
            self.root.after(2000, lambda: self.status.config(text="Your turn" if self.is_player_turn else "AI thinking..."))

    def _refresh(self):
        for sq in chess.SQUARES:
            label = self.square_labels[sq]
            piece = self.board.piece_at(sq)
            if piece:
                color = "white" if piece.color else "black"
                label.config(text=PIECE_UNICODE[piece.piece_type][color], font=self.piece_font)
            else:
                label.config(text="")

            base = self._base_color(self.config, sq)
            self.squares[sq].config(bg=base)
            label.config(bg=base)

        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                self.squares[sq].config(bg=self.config.last_move_color)
                self.square_labels[sq].config(bg=self.config.last_move_color)

        if self.selected_square is not None:
            self.squares[self.selected_square].config(bg=self.config.highlight_color)
            self.square_labels[self.selected_square].config(bg=self.config.highlight_color)
            for m in self.legal_from_selected:
                tgt = m.to_square
                if not self.board.piece_at(tgt):
                    self.square_labels[tgt].config(text="·", font=self.dot_font)

        self._update_eval_bar()
        self._update_fen_display()

    def _update_eval_bar(self):
        if self.game_over:
            return
        try:
            score, p_white, p_draw, p_black = self.inference.evaluate_position(self.board)
            score = max(-1.0, min(1.0, score))
            
            self.eval_text.config(text=f"{score:+.2f}")
            if score > 0.2:
                self.eval_text.config(fg="#4a90e2")
            elif score < -0.2:
                self.eval_text.config(fg="#e24a4a")
            else:
                self.eval_text.config(fg="white")
            
            self.prob_text.config(text=f"W:{p_white*100:.0f}% D:{p_draw*100:.0f}% B:{p_black*100:.0f}%")

            H = self.config.square_size * 8
            y = int((1.0 - score) * H / 2.0)
            y = max(0, min(H, y))
            self.eval_canvas.coords(self.eval_indicator, 0, y, 30, y)
        except Exception as e:
            print(f"[eval] error: {e}")

    # ---------- timer management ----------
    def _update_timer_display(self):
        """Update timer display every 100ms"""
        if self.clock is None:
            return
            
        # Update displays
        white_time = self.clock.get_display_time(chess.WHITE)
        black_time = self.clock.get_display_time(chess.BLACK)
        
        # Color code based on time remaining
        white_color = self._get_time_color(self.clock.white_ms)
        black_color = self._get_time_color(self.clock.black_ms)
        
        self.white_timer_label.config(text=white_time, fg=white_color)
        self.black_timer_label.config(text=black_time, fg=black_color)
        
        # Check for time forfeit
        if not self.game_over:
            if self.clock.is_flagged(chess.WHITE):
                self._time_forfeit(chess.WHITE)
            elif self.clock.is_flagged(chess.BLACK):
                self._time_forfeit(chess.BLACK)
        
        # Schedule next update
        if not self.game_over and self.clock:
            self.timer_job = self.root.after(100, self._update_timer_display)
        
    def _get_time_color(self, ms: int) -> str:
        """Get color based on time remaining"""
        if ms < 60000:  # < 1 minute
            return "#e24a4a"  # Red
        elif ms < 180000:  # < 3 minutes
            return "#f4a742"  # Orange
        return "white"
        
    def _time_forfeit(self, color: chess.Color):
        """Handle time forfeit"""
        winner = "Black" if color == chess.WHITE else "White"
        self.status.config(text=f"Time forfeit! {winner} wins!")
        self.game_over = True
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

    # ---------- input handlers ----------
    def _click_square(self, sq: int):
        if self.game_over or not self.is_player_turn:
            return
        piece = self.board.piece_at(sq)
        turn_color = chess.WHITE if self.player_is_white else chess.BLACK

        if self.selected_square is None:
            if piece and piece.color == turn_color:
                self.selected_square = sq
                self.legal_from_selected = [m for m in self.board.legal_moves if m.from_square == sq]
                self._refresh()
        else:
            promotion_moves = [m for m in self.legal_from_selected if m.to_square == sq]
            
            if promotion_moves:
                if len(promotion_moves) > 1:
                    chosen_promo = self._pick_promotion()
                    if chosen_promo is not None:
                        chosen = next((m for m in promotion_moves if m.promotion == chosen_promo), promotion_moves[0])
                        self._make_move(chosen)
                else:
                    self._make_move(promotion_moves[0])
            else:
                self.selected_square = None
                self.legal_from_selected = []
                if piece and piece.color == turn_color:
                    self.selected_square = sq
                    self.legal_from_selected = [m for m in self.board.legal_moves if m.from_square == sq]
            
            self._refresh()

    def _pick_promotion(self) -> Optional[int]:
        """
        Show promotion dialog with large, clearly visible buttons.
        FIXED: Uses wait_visibility() before grab_set() to prevent TclError
        Returns: chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, or None
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.resizable(False, False)
        dialog.configure(bg="#2b2b2b")
        
        # Make it modal
        dialog.transient(self.root)
        
        result = [None]

        # Title
        title = tk.Label(
            dialog, 
            text="Choose Promotion Piece:", 
            font=("Arial", 16, "bold"),
            bg="#2b2b2b",
            fg="white",
            pady=20
        )
        title.pack()
        
        # Button container
        button_frame = tk.Frame(dialog, bg="#2b2b2b")
        button_frame.pack(padx=20, pady=10)

        pieces = [
            (chess.QUEEN, "♕" if self.player_is_white else "♛", "Queen"),
            (chess.ROOK, "♖" if self.player_is_white else "♜", "Rook"),
            (chess.BISHOP, "♗" if self.player_is_white else "♝", "Bishop"),
            (chess.KNIGHT, "♘" if self.player_is_white else "♞", "Knight"),
        ]
        
        for ptype, symbol, name in pieces:
            # Create large, visible button
            btn = tk.Button(
                button_frame,
                text=f"{symbol}\n{name}",
                font=("Arial", 28, "bold"),
                width=5,
                height=3,
                bg="#4a4a4a",
                fg="white",
                activebackground="#5a5a5a",
                activeforeground="white",
                relief=tk.RAISED,
                bd=3,
                cursor="hand2",
                command=lambda pt=ptype: [result.__setitem__(0, pt), dialog.destroy()]
            )
            btn.pack(side=tk.LEFT, padx=8, pady=10)

        # Add default to Queen if dialog is closed without selection
        dialog.protocol("WM_DELETE_WINDOW", lambda: [result.__setitem__(0, chess.QUEEN), dialog.destroy()])
        
        # Center the dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # CRITICAL FIX: Wait for window to be visible before grabbing
        dialog.wait_visibility()
        
        try:
            dialog.grab_set()
        except tk.TclError as e:
            # Graceful fallback if grab fails
            print(f"Warning: Could not grab dialog: {e}")
        
        # Force focus
        dialog.focus_force()
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result[0] if result[0] is not None else chess.QUEEN

    # ---------- game flow ----------
    def _make_move(self, mv: chess.Move):
        # Stop clock for moving player
        if self.clock:
            moving_color = self.board.turn
            self.clock.stop_turn(moving_color)
        
        san = self.board.san(mv)
        self.board.push(mv)
        self.last_move = mv

        ply = len(self.board.move_stack)
        if ply % 2 == 1:
            self.history.insert(tk.END, f"{(ply + 1)//2}. {san} ")
        else:
            self.history.insert(tk.END, f"{san}\n")
        self.history.see(tk.END)

        self.selected_square = None
        self.legal_from_selected = []

        if self.board.is_checkmate():
            winner = "White" if not self.board.turn else "Black"
            self.status.config(text=f"Checkmate! {winner} wins!")
            self.game_over = True
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
                self.timer_job = None
        elif self.board.is_stalemate():
            self.status.config(text="Stalemate!")
            self.game_over = True
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
                self.timer_job = None
        elif self.board.is_insufficient_material():
            self.status.config(text="Draw – Insufficient material")
            self.game_over = True
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
                self.timer_job = None
        elif self.board.can_claim_draw():
            self.status.config(text="Draw available (threefold/50-move)")
            self.is_player_turn = not self.is_player_turn
        else:
            self.is_player_turn = not self.is_player_turn

        # Start clock for next player
        if self.clock and not self.game_over:
            next_color = self.board.turn
            self.clock.start_turn(next_color)

        if not self.game_over and not self.is_player_turn:
            self.status.config(text="AI thinking...")
            self.move_queue.put(True)
        elif not self.game_over:
            self.status.config(text="Your turn")

        self._refresh()

    def _ai_loop(self):
        while True:
            try:
                self.move_queue.get(timeout=0.1)
                if not self.game_over and not self.is_player_turn:
                    self._ai_play()
            except queue.Empty:
                continue

    def _ai_play(self):
        """AI move with optional MCTS"""
        start = time.perf_counter()
        
        try:
            if self.use_mcts and self.mcts_searcher:
                # Show progress UI
                self.root.after(0, self._show_mcts_progress)
                
                # Get time limit from clock (use max 30% of remaining time, min 1 second)
                time_limit_ms = None
                if self.clock:
                    ai_color = chess.BLACK if self.player_is_white else chess.WHITE
                    remaining_ms = self.clock.white_ms if ai_color == chess.WHITE else self.clock.black_ms
                    time_limit_ms = max(1000, int(remaining_ms * 0.3))
                
                # Run MCTS
                mv, stats = self.mcts_searcher.search(
                    self.board,
                    progress_callback=lambda c, t, e: self.root.after(0, self._update_mcts_progress, c, t, e),
                    time_limit_ms=time_limit_ms
                )
                
                print(f"[AI MCTS] move={mv} | nodes={stats['nodes_searched']} | "
                      f"value={stats['best_move_value']:+.3f} | visits={stats['best_move_visits']} | "
                      f"time={stats['total_time']:.2f}s")
                if stats['pv']:
                    print(f"[AI MCTS] PV: {' '.join(stats['pv'])}")
                
                self.root.after(0, self._hide_mcts_progress)
                
            else:
                # Direct policy
                mv, edge_conf, value_est = self.inference.predict_move(
                    self.board, use_top1=self.config.use_top1, temperature=self.config.temperature
                )
                print(f"[AI] move={mv} | edge_p={edge_conf:.3f} | eval={value_est:+.3f}")
            
            # Ensure minimum think time for UX
            elapsed = time.perf_counter() - start
            if elapsed < self.config.ai_think_time:
                time.sleep(self.config.ai_think_time - elapsed)
                
            self.root.after(0, self._execute_ai_move, mv)
            
        except Exception as e:
            print(f"[AI] error: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self._hide_mcts_progress)
            self.root.after(0, self._ai_error)

    def _execute_ai_move(self, mv: chess.Move):
        if mv in self.board.legal_moves:
            self._make_move(mv)
        else:
            self.status.config(text="AI made illegal move – You win!")
            self.game_over = True
            if self.timer_job:
                self.root.after_cancel(self.timer_job)
                self.timer_job = None

    def _ai_error(self):
        self.status.config(text="AI error – You win!")
        self.game_over = True
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

    def _new_game(self, color: Optional[str] = None):
        """Start new game with setup dialog"""
        # Cancel any ongoing timer
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None
        
        # Show setup dialog
        setup = GameSetupDialog(self.root)
        config = setup.show()
        
        if config is None:
            return  # Cancelled
            
        # Apply configuration
        color = config['color']
        if color == "random":
            color = "white" if (os.urandom(1)[0] % 2 == 0) else "black"
            
        # Setup clock
        if config['time_mode'] != 'none':
            self.clock = ChessClock(
                initial_ms=config['initial_time_min'] * 60 * 1000,
                increment_ms=config['increment_sec'] * 1000,
                delay_ms=config['delay_sec'] * 1000,
                mode=config['time_mode']
            )
        else:
            self.clock = None
            
        # Setup MCTS
        self.use_mcts = config['use_mcts']
        self.mcts_simulations = config['mcts_sims']
        if self.use_mcts:
            self.mcts_searcher = MCTSSearcher(self.inference, self.mcts_simulations)
        else:
            self.mcts_searcher = None
        
        # Reset game state
        if config.get('fen'):
            self.board = chess.Board(config['fen'])
        else:
            self.board = chess.Board()
        self.player_is_white = (color == "white")
        self.is_player_turn = self.player_is_white
        self.selected_square = None
        self.legal_from_selected = []
        self.game_over = False
        self.last_move = None
        
        self.history.delete(1.0, tk.END)
        self.status.config(text="Your turn" if self.is_player_turn else "AI thinking...")
        
        # Update timer displays
        if self.clock:
            self.white_timer_label.config(text=self.clock.get_display_time(chess.WHITE), fg="white")
            self.black_timer_label.config(text=self.clock.get_display_time(chess.BLACK), fg="white")
            # Start timer updates
            self._update_timer_display()
            # Start clock for first player
            start_color = self.board.turn
            self.clock.start_turn(start_color)
        else:
            self.white_timer_label.config(text="--:--", fg="white")
            self.black_timer_label.config(text="--:--", fg="white")
        
        if not self.is_player_turn:
            self.move_queue.put(True)
            
        self._refresh()

    def _undo_pair(self):
        if len(self.board.move_stack) >= 2 and not self.game_over and self.is_player_turn:
            self.board.pop()
            self.board.pop()
            
            self.history.delete(1.0, tk.END)
            tmp = chess.Board()
            for idx, mv in enumerate(self.board.move_stack, start=1):
                san = tmp.san(mv)
                if idx % 2 == 1:
                    self.history.insert(tk.END, f"{(idx + 1)//2}. {san} ")
                else:
                    self.history.insert(tk.END, f"{san}\n")
                tmp.push(mv)
            
            self.last_move = self.board.peek() if self.board.move_stack else None
            
            # Reset clock if exists
            if self.clock:
                # Cancel current timer
                if self.timer_job:
                    self.root.after_cancel(self.timer_job)
                    self.timer_job = None
                # Restart with current position
                self.clock.start_turn(self.board.turn)
                self._update_timer_display()
            
            self._refresh()

    def run(self):
        """Start the GUI and show initial setup dialog"""
        # Show setup dialog immediately
        self._new_game()
        
        # Bind window close to cleanup
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.root.mainloop()
    
    def _on_closing(self):
        """Cleanup before closing"""
        if self.timer_job:
            self.root.after_cancel(self.timer_job)
        self.root.destroy()


# ======================== MAIN ========================
def main():
    cfg = InferenceConfig()

    try:
        infer = ChessInference(run_dir=cfg.run_dir, checkpoint_path=cfg.checkpoint)
    except Exception as e:
        print(f"[inference] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    gui = ChessGUI(infer, cfg)
    gui.run()


if __name__ == "__main__":
    main()