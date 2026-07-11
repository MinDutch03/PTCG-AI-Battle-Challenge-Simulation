# PTCG AI Battle Challenge — Simulation Agent

Agent for [The Pokémon Company – PTCG AI Battle Challenge (Simulation)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle),
built on the [cabt engine](https://matsuoinstitute.github.io/cabt/) shipped with
`kaggle-environments`.

## How the agent plays

`main.py` exposes the Kaggle-required `agent(obs_dict) -> list[int]`. Decisions
flow through three layers, each a fallback for the one above, so the agent
never raises and never returns an illegal selection:

1. **Determinized Monte-Carlo turn search** (`ptcg_agent/search.py`).
   For every non-trivial selection we sample the hidden zones (our own deck +
   face-down prizes are exactly our 60-card list minus everything visible;
   opponent hidden cards are inert placeholders), replay the game into the
   engine's search sandbox (`search_begin`), then for each candidate root
   action roll the game forward with a heuristic policy **through the end of
   our turn and the opponent's reply turn** (`search_step`). Positions are
   scored by `ptcg_agent/evaluate.py`; the action with the best average over
   several determinizations wins. Because rollouts run inside the real engine,
   every card effect, evolution line, coin flip and prize trade is exact —
   no hand-written rules about individual cards.
2. **Heuristic policy** (`ptcg_agent/heuristics.py`). A complete rule-based
   policy covering every `SelectType`/`SelectContext` the engine can present.
   Used as the rollout policy inside the search, and as the direct answer
   when the clock is low or the search fails.
3. **First-legal-option fallback** — a last resort that needs nothing but the
   raw observation dict.

Supporting pieces:

- `ptcg_agent/determinize.py` — hidden-information sampling; pads prediction
  arrays with valid card IDs (native `SearchBegin` reads lengths from game
  state, so short arrays are a segfault risk).
- `ptcg_agent/evaluate.py` — position scoring: prizes ≫ damage-as-prize-equity
  ≫ board development ≫ hand/deck resources, with terminal states at ±1M.
- `ptcg_agent/cards.py` — thin cache over the engine's `all_card_data()` /
  `all_attack()`, so the agent needs no bundled card database.
- Time budgeting in `main.py` scales per-decision think time with the
  episode's `remainingOverageTime` (600 s pool) and hands the reins fully to
  heuristics when the clock runs low. An anti-loop guard wraps up the turn
  after 20 main-phase actions so repeatable abilities can't burn the clock.

## Deck

`deck.csv` is the submitted 60-card list. Candidate meta decks live in
`decks/` (Mega Abomasnow water, Archaludon ex, Dragapult ex, Hop's Trevenant
swarm, Crustle wall, Marnie's Grimmsnarl) — lists sourced from public
competition material and the observed July ladder meta, then evaluated
locally in round-robin tournaments under *this* pilot (`tools/tournament.py`),
because the best deck is the one the pilot plays best, not the best deck in
human hands.

## Ladder feedback loop

The Kaggle MCP server (`tools/kaggle_mcp.py`) pulls our submissions' episode
lists, replays and agent logs; `tools/analyze_replays.py` summarizes them
(opponent archetype, end reason, clock usage). Auditing real ladder losses
found the decisive pilot bugs (declined own-card effects, END chosen over a
powered attack, energy-less ex promoted as a shield) and produced the
dominance floors in `ptcg_agent/search.py`. Opponent decks harvested from
replays feed the inference library in `ptcg_agent/meta_decks/`. After the
fixes, the Mega Lucario matchup — 1-11 on the ladder, 35% of all losses —
retested at ~53% (n=36), with the ladder-weighted expected winrate at ~60%.

## Results (local, both sides piloted by this agent)

Deck selection was decided empirically. Six-deck round-robin at 10 games/pair
gave a shortlist; the top three replayed at 30 games/pair:

| Matchup (n=30 each) | Win rate |
|---|---|
| Grimmsnarl vs Dragapult | 70% |
| Grimmsnarl vs Trevenant | 60% |
| Grimmsnarl vs Archaludon | 70% |
| Dragapult vs Trevenant | 73% |

Marnie's Grimmsnarl ex is the submitted deck: no losing matchup found against
any tested archetype (5-5 vs Crustle wall and Mega Abomasnow at n=10), and it
beats the two most common ladder decks head-to-head. Against fixed baselines
with the same deck, the search pilot wins ~93% vs the heuristic-only policy
(n=30) and ~100% vs random.

Two decision-quality layers came out of auditing full game logs move by move:
a guaranteed-lethal fast-path (an action that wins in every sampled world is
taken immediately), and correct gust targeting — when a rollout picks the
opponent's forced new active (Boss's Orders), it drags their weakest,
highest-prize piece instead of reusing the "best fighter" ranking meant for
our own switches. Before that fix the pilot never played Boss's Orders
(0 of 11 opportunities in the audited game, two missed on-board wins);
after it, gust lines evaluate correctly and get played.

## Repo layout

```
main.py               Submission entry point (agent + deck selection)
deck.csv              Submitted deck (60 card IDs, one per line)
ptcg_agent/           Agent logic (search, heuristics, evaluation, sampling)
cg/                   Official cabt SDK: ctypes wrappers + engine binaries
                      (libcg.so is linux/x86-64 only — hence Docker below)
decks/                Candidate deck lists
baselines/            Random / greedy-heuristic opponents for evaluation
tools/                Local development harness (see below)
data/                 Card + attack database dumped from the engine
```

## Local development (macOS/anything via Docker)

The engine ships only a linux x86-64 `libcg.so`, so everything runs inside a
container:

```bash
# one-time
docker build --platform linux/amd64 -t cabt-dev tools
docker build --platform linux/amd64 -f tools/Dockerfile.kaggle -t cabt-kaggle tools

# play N games between two agent modules (seats alternate)
docker run --platform linux/amd64 --rm -v "$PWD":/work -w /work cabt-dev \
  python tools/run_match.py --p0 main --p1 baselines.greedy_agent -n 40

# same pilot, different decks — which deck does the agent play best?
docker run --platform linux/amd64 --rm -v "$PWD":/work -w /work cabt-dev \
  python tools/tournament.py -n 10

# build submission.tar.gz (main.py + deck.csv at archive root, per rules)
tools/package.sh decks/abomasnow.csv

# replicate Kaggle's validation episode (self-play through kaggle-environments)
docker run --platform linux/amd64 --rm -v "$PWD":/work -w /work cabt-kaggle \
  python tools/validate_submission.py
```

`tools/run_match.py` drives the engine directly through the SDK (no
kaggle-environments overhead): a full random-vs-random game runs in
milliseconds, search-pilot games in a few seconds.

## Design notes & non-obvious constraints

- **The search API is stateful C memory.** `search_begin` must be paired with
  `search_end` (we do it per determinization, in a `finally`); `SearchStep`'s
  id is an int64 — the ctypes signatures in `cg/sim.py` matter.
- **Determinism**: the engine places predicted hidden cards *in the order
  given* — the sampler must shuffle, or every "sample" is the same world.
- **Naive full-game PIMC is a known trap** (documented by other teams to
  *lower* ladder rating): with a garbage opponent model, deep lookahead
  hallucinates. We therefore only search to the end of the opponent's reply
  turn, where the opponent's *visible board* (real) dominates their hidden
  hand (guessed).
- **Kaggle sandbox quirks**: `__file__` can be undefined in the agent runner;
  deck.csv may live at `/kaggle_simulations/agent/`; both are handled in
  `main.py`. The validation episode is self-play — any crash fails the
  submission, hence the layered fallbacks.
- Timing: `actTimeout=0`, per-agent overage pool 600 s/episode. Games run
  100–200 decisions, so the budget tops out around ~1 s per decision and
  degrades gracefully.

## Submitting

```bash
tools/package.sh            # uses ./deck.csv
# upload submission.tar.gz under My Submissions
```

Only the latest 2 submissions stay in the matchmaking pool; ladder μ noise
between identical agents is large (±150 observed by other teams), so the final
candidate should occupy both slots.
