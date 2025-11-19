# ChessBert: A Tablebase + Stockfish-Trained Chess Transformer

A research project exploring whether master-level chess play can be achieved through supervised learning on engine games and tablebase positions, without human game data or AlphaZero-style massive tree search during training.

**Final Result:** A BERT-like transformer that plays at approximately **1500 Elo strength** through direct policy evaluation, trained on 6-piece tablebases and 100,000 Stockfish self-play games.

![bestaimlengineerv2](https://github.com/user-attachments/assets/fa058d5d-1a8c-4f73-9d4c-e30b28e1268b)

---

## Overview

This project investigates a fundamental question in chess AI: Can we train a strong chess player using only engine-generated data and tablebase positions, bypassing both human games and computationally expensive search-based training methods like AlphaZero?

**Key Features:**
- **BERT-like transformer** with 72-token position encoding
- **Multi-head architecture**: edge policy head (64×64 moves), promotion head, and value head (WDL)
- **Training data**: 6-piece Syzygy tablebases + 100,000 Stockfish long-time-control games
- **No human games** used in training
- **Direct policy play** achieved ~1500 Elo without tree search at inference
- **Interactive Tkinter GUI** with optional MCTS, time controls, and move analysis

**What Makes This Interesting:**
- Documents the **failure modes** of pure RL approaches (PPO self-play)
- Shows why MCTS-based training is **computationally prohibitive** for personal projects
- Demonstrates that **supervised learning on engine data** is vastly more efficient than pure RL
- Reveals that adding tree search can **hurt performance** if the value head isn't well-calibrated

---

## Motivation & Research Questions

This project was designed to explore several key questions:

1. **Can PPO self-play train a strong chess player from scratch?**
   - Without massive compute and tree search infrastructure like AlphaZero
   - Using only win/draw/loss rewards

2. **Is MCTS-driven training feasible for personal projects?**
   - Can we replicate AlphaZero-style training with consumer hardware?
   - How much compute would actually be required?

3. **Can we achieve strong play without human games?**
   - Using only engine self-play and perfect endgame positions (tablebases)
   - How does this compare to imitation learning on human games?

4. **What's the tradeoff between different approaches?**
   - Pure RL (PPO) vs supervised learning on engine games
   - Tablebase supervision vs engine supervision
   - Model-only inference vs adding tree search on top
     
---

## Project Timeline / Experiments

The project evolved through five distinct phases, each teaching valuable lessons about what works and what doesn't in chess AI.

### Phase 1: PPO Self-Play (Pure RL Attempt)

**Approach:**
- Train a chess agent from scratch using **Proximal Policy Optimization (PPO)**
- Pure reinforcement learning with self-play
- Reward signal: win (+1), draw (0), loss (-1)
- No imitation learning, no engine guidance, no tree search

**Hypothesis:**
- PPO could discover good chess play through trial and error
- Self-play would drive improvement through competitive pressure

**Reality:**
- **Complete failure** with available compute resources
- Chess has an enormous state space (10^40+ positions)
- Action space of ~200-300 legal moves per position
- Long game horizons (50-100 moves typical)
- Extremely sparse reward signals (only at game end)
- PPO struggled to learn anything beyond random play

**Key Lesson:**
> **PPO alone is not a practical way to train a strong chess player from scratch** without massive compute, tree search guidance, and sophisticated reward shaping. The combination of large state/action spaces, long horizons, and sparse rewards makes pure RL extremely sample-inefficient for chess.

### Phase 2: PPO + MCTS Training (Abandoned Due to Compute)

**Approach:**
- Use **Monte Carlo Tree Search** during training to generate better targets
- AlphaZero-style setup: MCTS uses the current network for evaluation
- Improved policy and value targets through search
- Train network to match MCTS-improved policies and values

**Why It Seemed Promising:**
- Combines the best of both worlds: RL exploration with tree search refinement
- Proven to work (AlphaZero, Leela Chess Zero)
- Should converge faster than pure self-play

**Reality Check:**
- Back-of-the-envelope calculations showed prohibitive compute requirements
- To reach master-level play (2200+ Elo) would require:
  - Millions of high-quality self-play games
  - Each requiring extensive MCTS (1000+ simulations per move)
  - Estimated **~2 years of continuous training** on available hardware (single GPU)
- This was far beyond the scope of a personal research project

**Key Lesson:**
> **MCTS-based training like AlphaZero requires industrial-scale compute resources.** While theoretically elegant, it's not accessible for personal projects or academic research without significant funding. The computational cost grows linearly with the desired strength level.

### Phase 3: Tablebases – 7-Piece Theory, 6-Piece Practicality, and PPO Finetuning

**Approach:**
- Use **Syzygy tablebases** for theoretically perfect endgame positions
- Tablebases provide exact solutions for positions with N or fewer pieces
- Extract training data: perfect move choices + WDL outcomes
- Try **PPO finetuning** on tablebase-derived targets

**Dataset Constraints:**
- 7-piece tablebases are massive (multiple terabytes) and impractical to download/process
- **6-piece tablebases** are more manageable (~140 GB) and were used instead
- Cover all endgames with 6 or fewer pieces on the board

**Data Construction:**
- For each 6-piece position:
  - **Policy target**: the single best move under perfect play
  - **Value target**: exact WDL (win/draw/loss) outcome with perfect play
- This gives theoretically perfect training data for the endgame

**PPO Finetuning Attempt:**
- Took a model trained on tablebase data
- Applied PPO to further refine the policy
- **Result: Made the model worse!**
  - PPO caused the policy to "average out" and become less precise
  - The model drifted away from crisp tablebase moves
  - Pre-finetuning model (pure supervised learning on tablebases) actually played better

**Key Lessons:**
> 1. **6-piece tablebases provide excellent endgame training data** and are practically downloadable/processable.
> 2. **PPO finetuning can harm performance** even when starting from perfect labels. The exploration/exploitation tradeoff in PPO can cause the model to drift away from optimal behavior.
> 3. **Supervised learning on perfect labels works better than RL-based refinement** in this context.

### Phase 4: Final Successful Approach – BERT-like Model on Tablebase + Stockfish Self-Play

**⭐ This is the main method that achieved the best results (~1500 Elo).**

**Data Sources:**
1. **6-piece Syzygy tablebases** – Perfect endgame positions
2. **100,000 Stockfish self-play games** – Long-time-control engine matches
   - Downloaded from official Stockfish testing archives
   - Provides realistic opening and middlegame positions
   - Each game records the final result (W/D/L)

**Training Targets:**

*Policy Head:*
- Predicts the **best move** in a given position
- Trained on either:
  - Perfect tablebase moves (for endgames)
  - Stockfish's chosen moves (for opening/middlegame)

*Value Head:*
- Predicts **Win-Draw-Loss probabilities** from White's perspective
- Trained on the **final game result** propagated back to each position
- Three-way classification: P(White wins), P(Draw), P(Black wins)
- The scalar evaluation is derived as: score = P(White wins) - P(Black wins)

**Model Architecture:**

A **BERT-style transformer** with bidirectional attention:

- **Input**: 72-token sequence encoding the chess position:
  - 1 [CLS] token
  - 64 square tokens (a1→h8, encoding piece type and color)
  - 1 side-to-move token (TURN_W or TURN_B)
  - 4 castling right tokens (WK/WQ/BK/BQ, YES or NO)
  - 1 en passant token (EP_A through EP_H, or EP_NONE)
  - 1 half-move clock bucket token (HMC_00_09 through HMC_90_99)

- **Architecture**:
  - 12 transformer layers
  - 12 attention heads per layer
  - 768 hidden dimensions
  - 3072 feed-forward dimensions
  - RoPE (Rotary Position Embeddings)
  - RMSNorm layer normalization
  - SwiGLU activations in feed-forward layers

- **Three Output Heads**:
  1. **Edge (Policy) Head**: Bilinear layer over 64×64 from-to moves
     - Projects square embeddings to lower dimension (256)
     - Computes bilinear similarity scores
     - Includes learned displacement bias based on from-to square distance
     - Output: probability distribution over 4,096 possible from-to edges
  
  2. **Promotion Head**: Disambiguates pawn promotions
     - Takes concatenated from/to square embeddings
     - MLP predicts promotion type (Queen/Rook/Bishop/Knight)
     - Only applies to moves that can be promotions
     - Output: 4-way classification per edge
  
  3. **Value Head**: MLP from [CLS] token
     - 2-layer MLP: 768 → 256 → 3
     - Output: 3-way WDL probabilities

**Training Details:**
- Trained for **9 epochs** on combined dataset
- Batch size: 64 positions
- Optimizer: AdamW (lr=3e-4, weight decay=0.05)
- Scheduler: Cosine annealing with warm restarts and linear warmup (3,000 steps)
- Loss: Multi-task loss combining:
  - Edge policy cross-entropy (label smoothing optional)
  - Promotion cross-entropy (gated to promotion moves only)
  - Value (WDL) cross-entropy
  - DeCov regularization on [CLS] representations
  - Orthogonality regularization on projection matrices and attention heads
- Hardware: Single GPU (CUDA required for reasonable training time)
- Training time: Approximately 8-12 hours per epoch on a modern GPU

**Results:**
- **~1500 Elo strength** in practical testing (ballpark estimate)
- Clean policy predictions with high confidence on strong moves
- Reasonable positional understanding from Stockfish game patterns
- Good endgame technique from tablebase training
- **This was by far the best-performing approach in the entire project**

**Why This Worked:**
- **Diverse, high-quality data**: Combines perfect endgames with strong middlegame/opening play
- **Supervised learning is efficient**: Direct optimization of known-good targets
- **Large dataset**: 100k games × ~40 positions/game = ~4 million positions
- **Strong prior from engine play**: Stockfish provides master-level examples to learn from

**Key Lesson:**
> **Supervised learning on high-quality engine data is vastly more efficient than pure RL** for chess. With limited compute, focus on data quality and straightforward supervised training rather than complex RL algorithms.

### Phase 5: Adding Tree Search on Top (MCTS Made It Worse!)

**Approach:**
- Take the trained value+policy model from Phase 4
- Add **Monte Carlo Tree Search** at inference time
- Use the model's policy head for move priors (which moves to explore)
- Use the model's value head as leaf evaluation (how good is this position)
- Standard MCTS algorithm: selection, expansion, evaluation, backpropagation

**Expectations:**
- Tree search should refine decisions beyond single-move policy predictions
- Deeper lookahead should catch tactical errors
- This is how most strong chess engines work (policy+value network + search)

**Reality:**
- **MCTS made the engine play worse!**
- Win rate decreased when MCTS was enabled
- Strength dropped noticeably in testing
- The pure policy+value model (without search) played better

**Why MCTS Failed Here:**

1. **Value Head Miscalibration:**
   - The value head was trained on limited data (~4M positions)
   - It wasn't reliable for positions several moves into the future
   - Accumulated value errors in deeper search trees

2. **Error Amplification:**
   - As MCTS searched deeper, small value errors compounded
   - The search effectively amplified the model's weaknesses rather than compensating for them

3. **Training-Inference Mismatch:**
   - Model was trained on "immediate next move" decisions
   - Not trained with search in the loop
   - Value head didn't account for being used as part of a tree search

**Contrast with AlphaZero:**
- AlphaZero trains with MCTS in the loop
- Value head learns to be reliable when used with search
- Much more training data and compute
- Value targets come from actual game outcomes after many moves of MCTS-selected play

**Key Lessons:**
> 1. **Tree search is not automatically beneficial.** A poorly calibrated value function can make search harmful rather than helpful.
> 2. **Training and inference should match.** If you want to use search at inference time, train with search in the loop.
> 3. **Value head quality matters enormously for search.** Without a well-calibrated value estimator, deeper search can amplify errors.
> 4. **Sometimes simpler is better.** The pure policy+value model was more reliable than the same model with search added.

---

## Model & Architecture

### Input Representation

The model uses a **72-token sequence** to represent any chess position:

```
Token Layout:
┌─────────────────────────────────────────────────────────────┐
│ [CLS] + 64 squares (a1→h8) + metadata (8 tokens)           │
└─────────────────────────────────────────────────────────────┘

Details:
- Token 0: [CLS] (classification token, aggregated for value head)
- Tokens 1-64: Square encodings (rank 1→8, file a→h within each rank)
  - Each square: piece type and color, or EMPTY
  - Example: "wP" (white pawn), "bN" (black knight), "EMPTY"
- Token 65: Turn indicator (TURN_W or TURN_B)
- Tokens 66-69: Castling rights (4 binary flags)
  - WK_CASTLE_YES/NO, WQ_CASTLE_YES/NO, BK_CASTLE_YES/NO, BQ_CASTLE_YES/NO
- Token 70: En passant file (EP_A through EP_H, or EP_NONE)
- Token 71: Half-move clock bucket (HMC_00_09, HMC_10_19, ..., HMC_90_99)
```

**Vocabulary Size:** ~60 tokens total
- 13 piece types (EMPTY + 6 white + 6 black pieces)
- Special fixed tokens (CLS, turn, castling flags, EP_NONE)
- 8 en passant files (A-H)
- 10 half-move clock buckets (0-9, 10-19, ..., 90-99)

### Transformer Architecture

**Core Design:**
- **Type:** BERT-style encoder with bidirectional attention
- **Layers:** 12 transformer blocks
- **Hidden size:** 768 dimensions
- **Attention heads:** 12 heads per layer (64 dimensions per head)
- **Feed-forward:** 3,072 dimensions with SwiGLU activation
- **Position encoding:** RoPE (Rotary Position Embeddings, base=10,000)
- **Normalization:** RMSNorm (more stable than LayerNorm)
- **Dropout:** 10% throughout

**Why These Choices:**
- **Bidirectional attention:** Chess positions are fully observable, no autoregressive requirement
- **RoPE:** Better generalization than learned positional embeddings
- **SwiGLU:** Empirically outperforms ReLU/GELU in transformers
- **RMSNorm:** Simpler and more stable than LayerNorm

### Output Heads

**1. Edge (Policy) Head:**

Projects the 64 square embeddings (from tokens 1-64) into a bilinear move space:

```python
# Conceptually:
from_proj = Linear(768 → 256)  # Project source square embeddings
to_proj = Linear(768 → 256)    # Project destination square embeddings
edge_scores = from_proj @ to_proj.T  # 64×64 bilinear scores
edge_scores += displacement_bias[Δfile, Δrank]  # Learned bias based on distance
edge_logits = reshape(edge_scores, 4096)  # Flatten to 4096 possible moves
```

- **Output shape:** [batch_size, 4096]
- **Interpretation:** Probability distribution over all possible from-to square combinations
- **Legal move masking:** Illegal moves are masked with -inf before softmax
- **Displacement bias:** Learned 15×15 table encoding distance-based move preferences

**2. Promotion Head:**

For moves that can be promotions, predicts the promotion piece type:

```python
# For each from-to edge:
concat = [from_embedding, to_embedding]  # 512 dims
promo_logits = MLP(concat → 128 → 4)     # 4 promotion types
```

- **Output shape:** [batch_size, 4096, 4]
- **4 types:** Queen (0), Rook (1), Bishop (2), Knight (3)
- **Gating:** Only applies to edges where promotion is legal
- **Legal promotion masking:** Only relevant promotion types are allowed per position

**3. Value Head:**

Predicts game outcome from the current position:

```python
cls_embedding = hidden_states[:, 0, :]  # [CLS] token
value_logits = MLP(cls_embedding → 256 → 3)  # Win/Draw/Loss
```

- **Output shape:** [batch_size, 3]
- **3 classes:** White wins (0), Draw (1), Black wins (2)
- **Evaluation score:** Derived as P(White win) - P(Black win)
- **Training target:** Final game result from each position's game

### Loss Function

Multi-task loss combining all objectives:

```
Total Loss = λ_edge × EdgeCE + λ_promo × PromoCE + λ_value × ValueCE + Regularizers

Where:
- EdgeCE: Cross-entropy on legal moves (masked, with optional label smoothing)
- PromoCE: Cross-entropy on promotion type (gated to promotion moves only)
- ValueCE: Cross-entropy on WDL classification
- Regularizers:
  - DeCov: Decorrelation of [CLS] representation dimensions
  - Orthogonality: Projection matrices should be near-orthogonal
  - Head diversity: Attention heads should attend to different patterns
```

**Loss weights:** edge=1.0, promo=1.0, value=1.0, decov_λ=1e-3, orth_λ=1e-3

---

## Data Pipeline

The data pipeline transforms raw Stockfish PGN archives into training-ready JSONL shards.

### Step 1: Download Stockfish Archives

**Script:** `download_stockfish_archives_to_100k.py`

```bash
python download_stockfish_archives_to_100k.py
```

**What it does:**
- Downloads long-time-control (LTC) Stockfish test games from Hugging Face
- Source: `official-stockfish/fishtest_pgns` dataset
- Downloads **newest games first** until reaching ≥100,000 total games
- Saves compressed PGN archives to `data/stockfish_ltc_raw/`
- Creates manifest file tracking downloads and game counts

**Output:**
- `data/stockfish_ltc_raw/*.pgn.gz` – Compressed game archives
- `data/stockfish_ltc_raw/_archives_manifest.jsonl` – Download tracking

**Note:** Only downloads what's needed. Resume-safe if interrupted.

### Step 2: Split Archives into Individual Games

**Script:** `split_stockfish_archives_to_games_100k.py`

```bash
python split_stockfish_archives_to_games_100k.py
```

**What it does:**
- Reads compressed PGN archives from Step 1
- Extracts each game into a separate `.txt` file
- Saves exactly **100,000 games** to `data/games/stockfish/`
- Each file named with timestamp, test ID, and game number
- Creates manifest with statistics

**Output:**
- `data/games/stockfish/*.txt` – One file per game (100,000 total)
- `data/games/stockfish/_manifest.json` – Processing summary

**File naming:** `YYYY-MM-DDTHH-MM-SSZ_ltc_chess_YY-MM-DD_testid_g1234.txt`

### Step 3: Convert Games to Training Shards

**Script:** `pgn_to_shards_white_pov.py`

```bash
python pgn_to_shards_white_pov.py
```

**What it does:**
- Reads all game files from `data/games/stockfish/`
- Processes **exactly 100,000 games** (stops after that)
- For **each move** in each game:
  - Encodes position as 72-token sequence
  - Records legal moves in that position
  - Records the move that was actually played (target)
  - Records final game result (W/D/L from White's POV)
- Performs **stable per-game split** into train/val shards
  - Hash of game metadata determines shard assignment
  - Same game always goes to same shard (no train/val leakage)
- Emits positions to JSONL shard files

**Output:**
```
data/shards/
├── shard_00.jsonl  ┐
├── shard_01.jsonl  │
├── shard_02.jsonl  │
├── shard_03.jsonl  ├─ Training shards (90% of games)
├── shard_04.jsonl  │
├── shard_05.jsonl  │
├── shard_06.jsonl  │
├── shard_07.jsonl  │
├── shard_08.jsonl  ┘
├── shard_09.jsonl  ← Validation shard (10% of games)
└── _pgn_to_shards_manifest.json
```

**JSONL format:**
```json
{
  "position_1d": ["CLS", "EMPTY", "EMPTY", ..., "HMC_00_09"],
  "legal_moves": ["Pe2e4", "Ng1f3", "Bf1c4", ...],
  "target": "Pe2e4",
  "wdl": 1
}
```

**Fields:**
- `position_1d`: 72 string tokens encoding the position
- `legal_moves`: All legal moves in modified UCI notation (e.g., "Pe2e4", "Pd7Qd8" for promotion)
- `target`: The move that was actually played
- `wdl`: Game result from White's perspective (+1 win, 0 draw, -1 loss)

**Statistics:**
- ~100,000 games × ~40 positions/game = **~4 million training positions**
- Split: ~3.6M training, ~400k validation

### Step 4: Tablebase Integration (Optional)

If you have 6-piece tablebases, you can generate additional endgame training data:

1. Download Syzygy 6-piece tablebases (~140 GB)
2. Extract positions where 6 or fewer pieces remain
3. Query tablebase for perfect move and WDL outcome
4. Create additional JSONL shards with perfect endgame labels
5. Mix with Stockfish shards during training (or train separately)

**Note:** The provided code focuses on the Stockfish game pipeline. Tablebase integration requires additional scripts not included in this repository, but the data format is identical.

---

## Training

### Base Training from Scratch

**Script:** `train.py`

**Prerequisites:**
1. Complete data pipeline (Steps 1-3 above)
2. CUDA-capable GPU strongly recommended
3. ~12 GB GPU memory for default batch size

**Configuration:** Edit `config.py` or modify defaults in script

**Key settings:**
```python
# Data
shards_dir = "data/shards"
train_shards = ["shard_00.jsonl", ..., "shard_08.jsonl"]  # 9 training shards
val_shard = "shard_09.jsonl"
batch_size = 64
num_workers = 4  # DataLoader workers

# Model
d_model = 768
n_layers = 12
n_heads = 12
d_ff = 3072
dropout = 0.10

# Optimization
lr = 3e-4
weight_decay = 0.05
warmup_steps = 3000
grad_clip = 1.0

# Training
save_every_epoch = 1
eval_every_epoch = 1
```

**Run training:**

```bash
python train.py
```

**Training behavior:**
- Cycles through training shards **infinitely** until interrupted
- Each epoch = one pass through a single shard
- Shards cycle in order: 00 → 01 → ... → 08 → 00 → ...
- Validation on shard_09 every N epochs
- Checkpoints saved every N epochs
- Press Ctrl+C to stop and save final checkpoint

**Output structure:**
```
runs/chess/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   ├── ...
│   └── checkpoint_latest.pt  ← Always points to most recent
├── logs/
│   └── train_YYYYMMDD_HHMMSS.log
├── plots/
│   ├── train_recent.png       ← Recent training metrics
│   └── val_history.png        ← Validation over epochs
├── metrics_train.csv
└── metrics_val.csv
```

**Monitoring progress:**

1. **Watch the terminal:** tqdm progress bars show smoothed metrics
   - L: Loss
   - E: Edge accuracy (top-1 move correct)
   - P: Promotion accuracy (on promotion moves only)
   - W: WDL accuracy (game outcome prediction)
   - LR: Current learning rate

2. **Check plots:** Updated after each validation
   - `plots/train_recent.png` – Recent training trends
   - `plots/val_history.png` – Validation metrics over time

3. **CSV files:** Complete metrics logs for analysis

**Training time:**
- ~1.5 hours per epoch on modern GPU (RTX 4070)
- ~9 epochs recommended for good performance
- Total: ~13.5 hours of training

**Recommended stopping point:**
- Monitor validation edge accuracy and WDL accuracy
- Stop when validation metrics plateau (typically 7-10 epochs)
- The model used for final results was trained for **9 epochs**

---

## Finetuning

### When to Finetune

Finetuning is useful when you have:
- A pretrained base model
- Additional specialized data (e.g., tactical puzzles, specific opening positions)
- Need to adapt the model without catastrophic forgetting

**Important:** In this project, PPO-based finetuning actually **hurt performance**. The finetuning script uses supervised learning with a lower learning rate and combines base data with finetune data to prevent forgetting.

### Finetuning Process

**Script:** `train_finetune.py`

**Prerequisites:**
1. Completed base training with saved checkpoint
2. Finetuning data in `data/shards_finetune/` (same format as base shards)

**Configuration:**

Edit `PRETRAINED_CHECKPOINT` in `train_finetune.py`:

```python
PRETRAINED_CHECKPOINT = "runs/chess/checkpoints/checkpoint_epoch_9.pt"
```

**Key differences from base training:**
- **Lower learning rate:** 3e-5 (10× lower than base training)
- **Constant LR:** No scheduler, prevents dramatic shifts
- **Combined datasets:** Each shard mixes base + finetune data
  - Prevents catastrophic forgetting
  - Maintains positional understanding while learning new patterns
- **Separate output directory:** `runs/chess_finetune/`

**Run finetuning:**

```bash
python train_finetune.py
```

**Finetuning behavior:**
- Loads pretrained model weights from checkpoint
- For each shard (00-09), combines:
  - Original base training data (`data/shards/shard_XX.jsonl`)
  - New finetuning data (`data/shards_finetune/shard_XX.jsonl`)
- Trains with gentle updates (low LR)
- Cycles infinitely until stopped
- Saves isolated checkpoints to avoid overwriting base model

**Output:**
```
runs/chess_finetune/
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   └── checkpoint_latest.pt
├── logs/
├── plots/
│   ├── train_recent_finetune.png
│   └── val_history_finetune.png
├── metrics_train.csv
└── metrics_val.csv
```

**Caution:**
- Monitor validation metrics carefully
- Finetuning can degrade performance if:
  - Learning rate is too high
  - Finetune data is low quality
  - Not enough mixing with base data
- The original Phase 3 experiments showed PPO finetuning hurt the model
- Pure supervised finetuning with dataset mixing works better

---

## Inference & Playing Against the Engine

### Running the Interactive GUI

**Script:** `inference.py`

**Prerequisites:**
1. Trained model checkpoint (from training or finetuning)
2. Python packages: tkinter, chess, torch

**Default configuration:**

The script loads from `runs/chess_finetune/checkpoints/checkpoint_latest.pt` by default.

To use a different checkpoint, edit `InferenceConfig` in `inference.py`:

```python
class InferenceConfig:
    run_dir: str = "runs/chess"  # Or "runs/chess_finetune"
    checkpoint: Optional[str] = None  # Or explicit path to .pt file
```

**Launch GUI:**

```bash
python inference.py
```

### GUI Features

**Game Setup Dialog (appears on startup):**

1. **Player Color:**
   - White: You play as White
   - Black: You play as Black
   - Random: Coin flip decides

2. **Time Control:**
   - **None:** Untimed game
   - **Fischer:** Time + increment per move (e.g., 5 minutes + 3 seconds)
   - **Delay:** Time with delay before clock starts (e.g., 5 minutes + 3 second delay)
   
3. **MCTS Settings:**
   - **Use MCTS:** Enable tree search (warning: may play worse!)
   - **Simulations:** Number of MCTS simulations per move (50-500 typical)

4. **Starting Position:**
   - Standard: Regular chess starting position
   - Custom FEN: Enter a specific position

**Main Interface:**

```
┌─────────────────────────────────────────┐
│  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜  ← 8               │
│  ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟  ← 7               │
│  · · · · · · · ·  ← 6               │
│  · · · · · · · ·  ← 5               │
│  · · · · · · · ·  ← 4               │
│  · · · · · · · ·  ← 3               │
│  ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙  ← 2               │
│  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖  ← 1               │
│  a b c d e f g h                     │
├─────────────────────────────────────────┤
│  White: 05:00  Black: 04:57          │  ← Time displays
│  Status: Your turn                   │  ← Game status
├─────────────────────────────────────────┤
│  1. e4 e5                             │
│  2. Nf3 Nc6                           │  ← Move history
│  3. Bb5 ...                           │
├─────────────────────────────────────────┤
│  [New Game] [Undo Pair] [Flip Board] │  ← Controls
└─────────────────────────────────────────┘
```

**Making Moves:**
1. Click source square to select piece
2. Legal destination squares highlight
3. Click destination to move
4. For promotions, dialog appears to select piece type
5. AI responds automatically after its thinking time

**Visual Feedback:**
- **Light green:** Selected square
- **Yellow:** Last move made (from and to squares)
- **Dark green:** Legal move destinations from selected piece

**Buttons:**
- **New Game:** Opens setup dialog, starts fresh game
- **Undo Pair:** Take back your last move and AI's last move
- **Flip Board:** Rotate board 180° (doesn't change who you're playing as)

### Inference Modes

**1. Pure Policy+Value (Recommended):**

Default mode. Model directly predicts best move:

```python
use_top1 = True  # Deterministic top-1 move
temperature = 0.2  # If use_top1=False, controls randomness
```

- **How it works:**
  1. Encode position to 72 tokens
  2. Forward pass through model
  3. Get edge logits over 4,096 possible moves
  4. Mask illegal moves with -inf
  5. Select top-1 move (or sample with temperature)
  6. If promotion, predict piece type from promotion head

- **Performance:** ~1500 Elo
- **Speed:** Instant (milliseconds per move)
- **Reliability:** Consistent strength

**2. Policy+Value + MCTS (Not Recommended):**

Adds tree search on top of model evaluation:

```python
use_mcts = True
mcts_simulations = 200  # More = slower but deeper search
```

- **How it works:**
  1. Root node = current position
  2. For each simulation:
     - Selection: Follow UCB1 tree policy to leaf
     - Expansion: Add legal moves as children
     - Evaluation: Run model on new position for value estimate
     - Backpropagation: Update visit counts and values up the tree
  3. After N simulations, select most-visited move

- **Performance:** Typically worse than pure policy (see Phase 5)
- **Speed:** Slow (seconds per move, depends on simulations)
- **Issues:** Value head miscalibration causes search to amplify errors

**Recommendation:** Use pure policy mode for best results. MCTS is included for experimental purposes but empirically performs worse.

### Model Outputs During Play

**Console logging:**

```
[AI] move=e2e4 | edge_p=0.847 | eval=+0.234
```

- `move`: Selected move in coordinate notation
- `edge_p`: Model's confidence in the move (policy probability after legal masking)
- `eval`: Position evaluation (positive = White better, negative = Black better)

**With MCTS enabled:**

```
[AI MCTS] move=e2e4 | nodes=200 | value=+0.156 | visits=87 | time=2.34s
[AI MCTS] PV: e2e4 e7e5 Ng1f3 Nb8c6 Bf1c4
```

- `nodes`: Number of MCTS simulations performed
- `value`: Average value estimate from tree search
- `visits`: Number of times this move was visited in search
- `PV`: Principal variation (best line found by search)

---

## Evaluation & Strength

### Strength Estimate: ~1500 Elo

**Methodology:**

The model's strength was estimated through practical testing against:
- Online play by going against bots from chess.com

**Results:**
- Begins to draw against intermediate engines or lose or win depending on different starting positions played. (~1500 Elo)


**Important caveats:**
- This is a **ballpark estimate**, not a rigorous rating
- Elo depends heavily on:
  - Time control (faster = weaker)
  - Opponent pool
  - Opening knowledge
- No formal rating through FIDE or established rating pools

### Playing Characteristics

**Strengths:**
- Strong in simplified endgames (tablebase training)
- Decent positional understanding (learned from Stockfish games)
- Doesn't blunder elementary tactics (usually)
- Fast move generation (no search latency)

**Weaknesses:**
- Limited tactical depth (no search = can't calculate forcing sequences)
- Occasional positional mistakes in complex middlegames
- Opening knowledge limited to Stockfish games seen in training
- Can miss long-term strategic plans
- Vulnerable to tactics requiring 3+ move calculations


## Lessons Learned

This project taught several valuable lessons about training chess engines and applying ML to complex strategy games.

### 1. Pure RL (PPO) Self-Play Is Extremely Hard

**Finding:** PPO without tree search, reward shaping, or strong priors cannot learn chess from scratch with reasonable compute.

**Why:**
- Enormous state space (~10^40 positions)
- Large action space (hundreds of legal moves)
- Very long horizons (50-100+ moves per game)
- Extremely sparse rewards (only at game end)
- Credit assignment problem (which move led to victory?)

**Implication:** Don't expect RL to "just work" on chess without massive infrastructure. AlphaZero needed:
- Thousands of TPUs
- Millions of games
- Sophisticated MCTS integration
- Months of training time

### 2. MCTS-Based Training Requires Industrial Compute

**Finding:** AlphaZero-style training (MCTS + RL) needs ~2 years of continuous single-GPU training to reach master level.

**Why:**
- Need millions of high-quality self-play games
- Each game requires 1000+ MCTS simulations per move
- Network inference is the bottleneck (even with GPU)
- Scales linearly with target strength

**Implication:** MCTS-based training is not accessible for:
- Academic research (without grants)
- Personal projects
- Hobbyist AI development
- Resource-constrained settings

**Alternative:** Focus on supervised learning with strong data sources.

### 3. Supervised Learning on Engine Data Is Vastly More Efficient

**Finding:** Training on 100k Stockfish games achieved better results in days than months of RL would have.

**Why:**
- Engine data is **information-dense**: every move is high-quality
- No wasted exploration in obviously bad positions
- Direct optimization of known-good behaviors
- Much faster convergence

**Comparison:**
```
Pure RL:         ??? games needed (didn't converge)
MCTS+RL:         Millions of games, years of compute
Supervised:      100k games, half a day of training → 1500 Elo ✓
```

**Implication:** When you have access to high-quality training data (engines, tablebases, expert games), use it! Supervised learning is orders of magnitude more sample-efficient than pure RL.

### 4. Data Quality Matters More Than Algorithm Complexity

**Finding:** Simple supervised learning on high-quality data beat sophisticated RL approaches.

**Principle:**
```
Strong data + simple algorithm > Weak data + complex algorithm
```

**Evidence from this project:**
- ❌ PPO self-play (complex RL, weak data) → Failed
- ❌ PPO + MCTS (complex RL+search, weak data) → Too expensive
- ❌ PPO finetuning (complex RL, perfect data) → Made model worse
- ✅ Supervised learning (simple, strong data) → Best results

**Implication:** Before reaching for complex RL algorithms:
1. Can you get high-quality training data? (engines, experts, simulations)
2. Can you frame this as supervised learning?
3. Have you exhausted simple approaches?

Only use RL when:
- No good supervised data exists
- Exploration is necessary
- You have sufficient compute resources

---

## Repository Structure

```
.
├── README.md                    # This file
│
├── config.py                    # Configuration dataclasses (model, data, training)
├── model.py                     # ChessEdgeModel (BERT-like transformer)
├── loss.py                      # Multi-task loss function
├── metrics.py                   # Accuracy calculations
├── data_loader.py               # JSONL dataset & DataLoader
├── checkpoint.py                # Checkpoint saving/loading
├── scheduler.py                 # Learning rate scheduler (cosine with restarts)
├── logger.py                    # Logging utilities
├── plots.py                     # Metrics plotting
├── utils.py                     # Helper functions
│
├── train.py                     # Main training script
├── train_finetune.py            # Finetuning script (combines base + finetune data)
├── inference.py                 # Interactive GUI for playing vs the model
│
├── download_stockfish_archives_to_100k.py    # Step 1: Download Stockfish games
├── split_stockfish_archives_to_games_100k.py # Step 2: Extract individual games
├── pgn_to_shards_white_pov.py                # Step 3: Convert to JSONL shards
│
├── data/
│   ├── stockfish_ltc_raw/       # Downloaded PGN archives
│   ├── games/
│   │   └── stockfish/           # Individual game files (100k .txt files)
│   ├── shards/                  # Training JSONL shards
│   │   ├── shard_00.jsonl       # Training
│   │   ├── ...
│   │   ├── shard_08.jsonl       # Training
│   │   └── shard_09.jsonl       # Validation
│   └── shards_finetune/         # Optional finetuning shards
│
└── runs/
    ├── chess/                   # Base training outputs
    │   ├── checkpoints/         # Model checkpoints (.pt files)
    │   ├── logs/                # Training logs
    │   ├── plots/               # Metrics plots (PNG)
    │   ├── metrics_train.csv    # Training metrics
    │   └── metrics_val.csv      # Validation metrics
    └── chess_finetune/          # Finetuning outputs (same structure)
```

**Key files:**
- `config.py` → All hyperparameters and settings
- `model.py` → Transformer architecture
- `train.py` → Main training loop
- `inference.py` → Play against the model
- `pgn_to_shards_white_pov.py` → Data preprocessing

**Data flow:**
```
Raw PGN archives → Individual games → JSONL shards → Training → Checkpoint → Inference
```

---

## Setup & Requirements

### System Requirements

**Hardware:**
- **GPU:** CUDA-capable GPU strongly recommended
  - Minimum: 8 GB VRAM (will be slow)
  - Recommended: 12+ GB VRAM (RTX 3090, A100, etc.)
  - CPU-only training is possible but extremely slow (not recommended)
- **RAM:** 16 GB+ system RAM
- **Storage:** 
  - ~150 GB for Stockfish archives and extracted games
  - ~10 GB for training shards
  - ~5 GB for model checkpoints and logs

**Software:**
- **Python:** 3.8+
- **Operating System:** Linux (recommended), macOS, or Windows
  - Linux: Best multiprocessing performance
  - Windows: Works but DataLoader may be slower

### Python Dependencies

**Core libraries:**
```txt
torch>=2.0.0           # PyTorch with CUDA support
chess>=1.10.0          # python-chess for chess logic
tqdm>=4.65.0           # Progress bars
matplotlib>=3.7.0      # Plotting
huggingface-hub>=0.16.0  # Downloading Stockfish archives
```

**GUI (for inference.py):**
```txt
tkinter                # Usually included with Python
```

**Installation:**

```bash
# Clone repository
git clone https://github.com/rsenatorov/ml-chess-bert.git
cd ml_chess_bert

# Install dependencies
pip install torch chess tqdm matplotlib huggingface-hub

# Verify CUDA available (should print True)
python -c "import torch; print(torch.cuda.is_available())"
```

**For CUDA support:**
- Install PyTorch with CUDA from: https://pytorch.org/get-started/locally/
- Example (CUDA 11.8):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

### Quick Start

**1. Prepare data:**
```bash
# Download Stockfish archives (~100k games, will take time depending on connection)
python download_stockfish_archives_to_100k.py

# Extract individual game files (may take 15-30 minutes)
python split_stockfish_archives_to_games_100k.py

# Convert to training shards (may take 30-60 minutes)
python pgn_to_shards_white_pov.py
```

**Expected output:**
- `data/shards/shard_00.jsonl` through `shard_09.jsonl`
- Total ~4 million positions

**2. Train model:**
```bash
# Start training (will run indefinitely until stopped)
python train.py

# Monitor in real-time:
# - Terminal: tqdm progress bars with metrics
# - Plots: runs/chess/plots/*.png (updated each validation)
# - Logs: runs/chess/logs/*.log
# - CSVs: runs/chess/metrics_*.csv

# Stop with Ctrl+C when validation metrics plateau (typically 7-10 epochs)
# Checkpoint will be saved automatically
```

**3. Multi-GPU training:**

The current code doesn't include DataParallel/DistributedDataParallel, but you can add it:

```python
# In train.py, wrap model:
model = torch.nn.DataParallel(model)
```

**4. Logging and debugging:**

Increase verbosity:
```python
# In logger.py
get_logger("train", logs_dir, level=logging.DEBUG)
```

---

## Acknowledgements

**Data Sources:**
- [**Stockfish**](https://stockfishchess.org/) – The strongest open-source chess engine
  - Fishtest LTC games used for training data
  - Stockfish developers and contributors
- [**Syzygy Tablebases**](https://syzygy-tables.info/) – Perfect endgame databases
  - Created by Ronald de Man
  - 6-piece tablebases used for endgame training

**Libraries:**
- [**PyTorch**](https://pytorch.org/) – Deep learning framework
- [**python-chess**](https://python-chess.readthedocs.io/) – Chess logic and PGN parsing
- [**Hugging Face Hub**](https://huggingface.co/docs/huggingface_hub/) – Dataset downloads

**Inspiration:**
- **AlphaZero** (DeepMind) – Pioneer of RL + search for chess/go
- **Leela Chess Zero** – Open-source AlphaZero implementation
- **Maia Chess** – Human-like chess AI research

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@software{chess_bert_2025,
  title={ChessBert: A Tablebase and Stockfish-Trained Chess Transformer},
  author={Robert Senatorov},
  year={2025},
  url={https://github.com/rsenatorov/ml-chess-bert.git}
}
```

---

## License

See attached license.

---

## Contact

Feel free to connect with me on LinkedIn: www.linkedin.com/in/robert-senatorov-a09753303
