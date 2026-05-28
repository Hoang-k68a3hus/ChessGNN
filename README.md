# ChessGNN

ChessGNN is a research-style neural chess engine built around a heterogeneous
graph representation of the chess board. The project explores a three-phase
training pipeline:

1. Supervised policy/value learning from games.
2. A latent world model with VAE, transition dynamics, diffusion refinement, and
   EBM-based legality scoring.
3. Self-play reinforcement learning with replay, latent search, and optional
   MCTS-style planning.

The goal is not to wrap Stockfish or hand-code chess heuristics. The goal is to
learn chess decisions from structured board relations: pieces, squares, legal
moves, attacks, defenses, pins, king zones, pawn structure, and game history.

## Demo

<video src="docs/demo/lichess_chessgnn_demo.mp4" controls width="100%"></video>

If the video does not render in your browser, open it directly:
[docs/demo/lichess_chessgnn_demo.mp4](docs/demo/lichess_chessgnn_demo.mp4)

[![ChessGNN Lichess demo thumbnail](docs/demo/lichess_chessgnn_demo.jpg)](docs/demo/lichess_chessgnn_demo.mp4)

Demo: ChessGNN checkpoint playing as White against Lichess AI Level 6 through
the Lichess bot adapter.

The demo video shows ChessGNN running as a live Lichess bot through the
Lichess bot bridge used during local development. The bot loads a PyTorch
checkpoint, converts the current Lichess board to a graph with `python-chess`,
masks illegal moves, and selects a move from the model policy head. The local
development version also supports optional latent MCTS-style search.

Current qualitative result: in informal personal testing, the Phase 1
policy/value checkpoint consistently beat Stockfish/Lichess AI Level 5 and
remained competitive against Level 6 through the middlegame, while still
struggling in endgames. Formal Elo evaluation is pending; evaluation tooling was
built locally to make future comparisons reproducible and balanced by color and
Lichess AI level.

## Motivation / Problem

Most neural chess projects start with an 8x8 tensor and use a CNN. That is a
reasonable baseline, but chess is not just image-like local texture:

| Chess property | CNN board tensor | ChessGNN graph |
| --- | --- | --- |
| Long-range sliding attacks | Must be learned indirectly through many layers | Direct ATTACK, CONTROL, XRAY, and PIN edges |
| Legal move structure | Usually added as a mask after prediction | Legal moves are first-class graph edges |
| Piece interactions | Encoded as channels on squares | Explicit source/destination edge relations |
| Non-local tactics | Hard for small kernels | Message passing follows tactical relations |
| Chess-specific rules | Need extra channels or post-processing | Castling, en passant, king zone, pawn structure edges |
| Game history | Not captured by a single board image | Causal Transformer over position sequence |

The core idea is to represent a chess position as a heterogeneous directed graph:

- 64 fixed nodes, one per square.
- 29 node features per square.
- Dynamic directed edges for legal moves, attacks, defenses, pins, x-rays, king
  zones, pawn structure, castling, en passant, and control.
- 27 edge features per relation.
- 10 semantic edge types.

This lets the model perform message passing over the same relations a chess
player reasons about, instead of forcing everything through local image filters.

## Architecture Overview

### Phase 1 - Supervised Policy/Value Model

Phase 1 trains the playable chess agent. It predicts:

- `value`: scalar game outcome estimate for each position.
- `policy`: 64 x 64 from-square/to-square move logits.

```mermaid
flowchart LR
    A[PGN / FEN board] --> B[python-chess Board]
    B --> C[Heterogeneous ChessGraph]
    C --> C1[64 square nodes<br/>29 node features]
    C --> C2[Dynamic directed edges<br/>10 edge types<br/>27 edge features]
    C1 --> D[Edge-Conditioned GATv2 Encoder]
    C2 --> D
    D --> E[Per-square embeddings<br/>B x T x 64 x d_node]
    D --> F[Attentive graph readout<br/>B x T x d_node]
    F --> G[pos_to_z projection<br/>d_node -> d_z]
    G --> H[Causal Game Transformer<br/>RoPE + Flash Attention]
    E --> I[Square Cross-Attention]
    H --> I
    I --> J[Game embedding<br/>B x T x d_z]
    J --> K[Value head<br/>B x T x 1]
    J --> L[Policy head<br/>B x T x 64 x 64]
    E --> L
    M[Legal move mask] --> L
```

Important implementation choices:

- `board_to_graph()` builds a typed graph from `python-chess`.
- `ChessGNNEncoder` projects raw node/edge features into learned embeddings.
- `EdgeCondGATv2Layer` injects edge embeddings directly into attention scores.
- `GameLevelEncoder` batches all `B*T` positions into one disconnected graph
  batch, avoiding a Python loop over timesteps.
- `GameTransformer` models history with causal self-attention and RoPE.
- `SquareCrossAttention` brings square-level spatial features back into the
  temporal game embedding before the heads.
- `PolicyHead` scores every from-square/to-square pair and applies a legal move
  mask before inference.

Default dimensions in the current model:

| Component | Value |
| --- | --- |
| Raw node feature dim | 29 |
| Raw edge feature dim | 27 |
| Node embedding dim `d_node` | 128 |
| Edge embedding dim `d_edge` | 64 |
| Latent/game dim `d_z` | 256 |
| GNN layers | 6 |
| Transformer layers | 6 |
| Policy space | 4096 from-to logits |

### Phase 2 - Latent World Model

Phase 2 extends the supervised model into a world model. The purpose is to learn
a latent state space where the engine can reconstruct positions, predict future
states, refine graph embeddings, and score legality.

```mermaid
flowchart TD
    A[Phase 1 encoder output] --> B[game_emb<br/>B x T x d_z]
    A --> C[h_node_seq / h_edge_flat<br/>graph embeddings]
    B --> D[VAE posterior<br/>mu, logvar]
    D --> E[z sequence]
    E --> F[Graph VAE decoder]
    F --> G[Reconstructed node features<br/>and edge predictions]
    E --> H[Transformer transition model]
    H --> I[z_t -> z_t+1 prediction]
    C --> J[Conditional graph diffusion]
    E --> J
    J --> K[Denoised node/edge embeddings]
    K --> L[EBM legality scorer]
    L --> M[Lower-energy legal graph states]
    E --> N[Value/policy heads in z-space]
```

Phase 2 is intentionally split into smaller stages to reduce gradient conflicts
and memory pressure:

| Stage | Goal | Main modules |
| --- | --- | --- |
| 2A | Build stable latent space | VAE posterior, graph decoder, z value/policy heads |
| 2B | Learn dynamics | Transformer transition model |
| 2C | Sharpen graph embeddings | Conditional diffusion denoiser |
| 2D | Joint refinement | Diffusion + EBM legality scoring |

Training safeguards in the codebase include:

- KL warmup and free-bits to reduce posterior collapse.
- Component-level loss weights for reconstruction, transition, diffusion, EBM,
  value, policy, and anchor losses.
- Optional loss normalization by moving average.
- Separate optimizers for VAE, transition, diffusion, EBM, and encoder groups.
- Encoder freeze/unfreeze controls for Phase 2.
- FP32 branch support for diffusion/EBM when AMP instability is a risk.
- Split-stage checkpoints and quality-gate metrics.

Phase 2 is active research work in this repository. The implementation is meant
to expose the full modeling stack and training infrastructure, not to claim a
finished world-model chess engine yet.

### Phase 3 - Self-Play RL and Search

Phase 3 is the reinforcement learning layer. It uses the supervised model and
latent world model as a foundation, then improves policy/value estimates through
self-play trajectories.

```mermaid
flowchart LR
    A[Current checkpoint] --> B[Self-play game generator]
    B --> C[LatentSearch or LatentMCTS]
    C --> D[Move policy / root value]
    D --> E[Game trajectory]
    E --> F[Replay buffer]
    F --> G[Phase3SelfPlayTrainer]
    G --> H[Policy/value update]
    G --> I[Transition fine-tuning]
    G --> J[Hierarchical temporal module]
    G --> K[Adversarial latent dynamics]
    H --> A
```

Key components:

- `ReplayBuffer` stores positions, policies, values, and trajectories.
- `LatentSearch` evaluates candidate moves in latent space.
- `LatentMCTS` supports AlphaZero-style visit-count search.
- `ParallelSelfPlayManager` runs multi-process self-play workers with a batched
  GPU inference server.
- `HierarchicalGameTransformer` models multiple temporal resolutions.
- `AdversarialDynamics` explores white/black latent policies and Nash-style
  objectives.

Phase 3 is experimental. The project already contains the training and search
infrastructure, but the strongest stable public-facing story is still the Phase
1 supervised GNN + Transformer agent plus the Phase 2 world-model extension.

## Data Flow

For supervised training, a PGN game becomes a sequence of graph positions:

```text
PGN game
  -> FullGameDataset
  -> [B, T] graph sequence with padding mask
  -> batched disconnected ChessGraph for all B*T positions
  -> GNN node/edge embeddings
  -> causal temporal Transformer
  -> value and legal-masked policy heads
```

The policy target is indexed as:

```python
policy_index = move.from_square * 64 + move.to_square
```

The model outputs a dense 4096-way policy, then masks it with the legal move
matrix from `python-chess`.

## Implementation Scope

The local implementation is organized around these components:

| Component | Role |
| --- | --- |
| Graph representation | Converts a board into 64 square nodes and typed chess edges |
| Edge-conditioned GATv2 encoder | Performs relational message passing over legal moves, attacks, defenses, pins, and control |
| Full-game encoder | Batches all positions in a game and produces square-level and position-level embeddings |
| Causal game Transformer | Models move history with RoPE positional encoding |
| Square cross-attention | Re-injects per-square spatial information before policy/value prediction |
| VAE world model | Learns a latent state distribution and reconstructs graph structure |
| Transition Transformer | Predicts future latent states |
| Diffusion decoder | Refines node/edge graph embeddings |
| Energy model | Scores graph states for legality correction experiments |
| Self-play/search layer | Uses replay, latent search, and optional MCTS-style planning |
| Lichess adapter | Connects a trained checkpoint to live online play |

## Demo Setup Summary

The Lichess demo was produced with a local adapter around `lichess-bot`:

1. Lichess sends the current board state to the local bot.
2. The adapter converts the position to a `python-chess` board.
3. ChessGNN builds the heterogeneous graph representation.
4. The policy/value checkpoint scores all 4096 from-square/to-square moves.
5. Illegal moves are masked.
6. The highest-scoring legal move is sent back to Lichess.

The public video is a compressed artifact of that workflow. It is meant as a
readable HR/portfolio demo, not as an Elo benchmark.

## Evaluation Plan

Before making any formal strength claim, the evaluation should use:

- At least 40 games per checkpoint.
- Balanced white/black games.
- Fixed Lichess AI level distribution.
- Raw score and bucket-balanced score.
- Separate reporting for opening/middlegame quality and endgame failures.

This is why the README only makes a qualitative demo claim rather than a formal
rating claim.

Full training code and checkpoints are available on request.

## Current Status and Limitations

What is solid:

- Heterogeneous graph representation for chess positions.
- Edge-conditioned GATv2 encoder.
- Full-game batched supervised policy/value training.
- Causal Transformer over game history.
- Legal-move masking and Lichess bot integration.
- Training infrastructure for multi-component world-model experiments.

What is still in progress:

- Stable Phase 2 world-model training across all components.
- Statistically meaningful Elo evaluation.
- Endgame strength.
- Fully validated Phase 3 self-play RL.
- Production-ready model packaging.

Known modeling risks:

- Phase 2 has multiple objectives with different loss scales.
- VAE KL can dominate if warmup/free-bits are poorly tuned.
- EBM negatives can become too easy if corruption is unrealistic.
- Self-play search must match the same encoder path used during training.
