# Sana 🪼 — 100M Parameter Edition

A ~100M parameter language model trained from scratch. No Hugging Face transformers. No SentencePiece. No flash-attn. Every component hand-implemented in pure PyTorch. Scale-up from the 24M edition.

---

## Architecture

| Component        | Detail                                          |
| ---------------- | ----------------------------------------------- |
| Parameters       | ~98.75M unique (lm_head weight tied to embedding) |
| Hidden dim       | 640                                             |
| Layers           | 18 TransformerBlocks                            |
| Attention heads  | 10                                              |
| Head dim         | 64 (640 / 10) — same as 24M                     |
| FFN intermediate | 1,707 (640 × 2.667 rounded, SwiGLU)             |
| Vocabulary       | 16,000 tokens (BPE, pure Python) — same tokenizer as 24M |
| Max sequence     | 512 tokens                                      |
| Position encoding| RoPE (Rotary Position Embeddings), base=10000   |
| Normalisation    | RMSNorm (pre-norm architecture)                 |
| Activation       | SwiGLU (gated FFN)                              |
| KV Cache         | Yes (per-layer, allocated during generation)    |
| Tied embeddings  | Yes (input/output share one weight matrix)      |
| Dropout          | 0.1 during pretrain, 0.0 during SFT             |

**Why `num_heads=10`:** gives `head_dim=64`, a clean power of 2, same as the 24M model. Attention math is identical — only width and depth changed.

---

## Special Tokens

Identical to the 24M model. Tokenizer is fully shared.

| ID  | Token            | Purpose                   |
| --- | ---------------- | ------------------------- |
| 0   | `<\|pad\|>`      | Padding                   |
| 1   | `<\|user\|>`     | User turn marker          |
| 2   | `<\|sana\|>`     | Sana turn marker          |
| 3   | `<\|end\|>`      | End of turn / EOS         |
| 8   | `<sana_salute>`  | Sana emotion: 🪼 greeting |
| 9   | `<sana_happy>`   | Sana emotion: ✨ happy    |
| 10  | `<sana_think>`   | Sana emotion: 🤔 thinking |
| 11  | `<sana_sad>`     | Sana emotion: 😔 sad      |
| 12  | `<sana_smug>`    | Sana emotion: 😏 smug     |

---

## What It Can Do

### Works well

**Factual questions on topics covered by FineWeb-Edu**
```
You: what is a black hole
Sana: 🤔 Black holes are regions in spacetime where gravity is so strong that nothing,
      not even light, can escape. They're formed from collapsing stars.

You: explain photosynthesis
Sana: 🤔 Photosynthesis is the process by which plants convert sunlight, carbon dioxide,
      and water into glucose and oxygen. It's basically the way we use energy.

You: what is nuclear fusion
Sana: 🤔 It's the process where light collapses into energy, releasing vast amounts of
      energy. Pretty fascinating.
```

**Emotional routing**
```
You: im sad
Sana: 😔 That's rough. What's going on?

You: i got the job!!
Sana: ✨ Nice one. When do you start?

You: cool!
Sana: ✨ Glad you think so!
```

**Identity**
```
You: who are you
Sana: 😏 I'm Sana. Compact AI. Short answers, dry delivery.

You: tell me your name
Sana: 🤔 Sana. What do you need?

You: what do you like
Sana: ✨ Jellyfish. Octopuses. Black holes. Things that shouldn't work but do.
```

**Graceful degradation on out-of-domain queries**
```
You: how do i cook a potato
Sana: 🤔 You can use my own recipe or mix it with some spice. Your choice.

You: what is a nintendo switch
Sana: 🤔 A switch for moving parts. Sounds familiar?
```
The model doesn't know consumer electronics but infers from semantic context rather than hallucinating a different topic. It knows "Nintendo Switch" is a thing, it just doesn't know what kind of thing specifically.

**Jellyfish (Sana's speciality)**
```
You: tell me about jellyfish
Sana: 🤔 Jellyfish have been around for over 500 million years. They're basically immortal.
      Their tentacles can revert to their juvenile form in the ocean.

You: anything else?
Sana: ✨ Jellyfish have been around for over 500 million years. They're basically the
      ultimate survivors.
```

---

## Known Failures and Limitations

### Entity tracking — not reliable
The model was trained on only 8 handcoded entity-tracking examples. It cannot consistently remember names introduced earlier in a conversation.

```
You: my cat is named Lumi
Sana: 😔 That's cool! What's going on?

You: what is my cat's name
Sana: 🤔 I'm Sana. Hi, question. How can I find you?   ← forgets Lumi
```

**Fix for next iteration:** expand HC_ENTITY to 50+ examples covering pets, people, and self-introduction across 3-4 turn conversations.

### Greeting token misfires
The model correctly produces greeting-appropriate content but sometimes uses the wrong emotion token. `"hi"` occasionally produces ✨ instead of 🪼. The content routing is correct; only the emoji is wrong.

```
You: hi
Sana: ✨ Hey. What do you need?   ← should be 🪼, content is right
```

**Root cause:** greeting responses (avg 5 tokens) have ~4× less gradient signal per example than empathy responses (avg 20 tokens). Fix requires first-token loss weighting in `finetune.py`.

### Consumer / niche knowledge gaps
FineWeb-Edu is educational text. The model has weak or absent knowledge of:
- Consumer electronics (phones, gaming consoles, specific products)
- Pop culture, celebrities, recent events
- Cooking, recipes, practical how-to tasks
- Sports, music, entertainment
- Anything post-training-cutoff

The model degrades gracefully — it gives a plausible-sounding response based on semantic context rather than outputting garbage. But the responses are vague.

### Numerical precision
The pretrain data contained unicode artifacts (`e28099`, `\x10`, `\x12`) from pre-unicode-fix downloads. Some numerical outputs contain garbled characters.

```
You: what is the speed of sound
Sana: 🤔 Sound can travel at about 1/)th that of a vacuum.   ← \x00 artifact
```

**Fix:** retrain pretrain on clean data (unicode fix was applied to `download_data.py` and `tokenize_data.py` but the 100M pretrain used older-downloaded shards).

### Multi-turn context bleeding
With `max_history_turns=3`, responses from several turns ago can influence current responses in unexpected ways. Set `max_history_turns=1` in `inference.py` to mitigate.

### Factual confabulation
The model produces confident-sounding but wrong answers, especially on specific facts (dates, numbers, proper names).

```
You: who is aristotle
Sana: 🤔 Aristotles are ancient Greek philosophers with profound knowledge and wisdom.
      Their works have been around for over 500 years.   ← 2,400 years, not 500
```

At 100M parameters trained on 7.8B tokens this is expected. RAG is stubbed out in `inference.py` for future implementation.

---

## The Bioluminescence Experiment 🔬

Same experiment as the 24M edition. Demonstrates that pretrain knowledge survives SFT.

`"bioluminescence"` does not appear anywhere in `data/sft_direct.jsonl`. Ask the model anyway:

```bash
grep -i "bioluminescen" data/sft_direct.jsonl && echo "FOUND" || echo "CLEAN"
python inference/inference.py --checkpoint checkpoints/finetune
# → What is bioluminescence?
```

Expected: a response describing light from chemical reactions in living organisms, using `<sana_think>`, despite never appearing in SFT data. Passes consistently at epoch 3.

---

## Pretrain vs 24M Comparison

| Metric                  | 24M         | 100M          |
| ----------------------- | ----------- | ------------- |
| Parameters              | 23.85M      | 98.75M        |
| Pretrain tokens         | 2.09B       | 7.84B         |
| Pretrain val loss       | 2.048       | 2.053         |
| Pretrain test score     | 68–76%      | 72–84%        |
| SFT examples            | ~1,200      | ~9,690        |
| SFT epochs (sweet spot) | 5           | 3             |
| SFT optimizer steps     | ~375        | ~1,800        |
| SFT lr                  | 1.5e-5      | 1.0e-5        |
| Best SFT test score     | 81.8%       | 90.9%         |

The 100M pretrain val loss (2.053) matches the 24M (2.048) despite 4× more parameters because the pretrain data still contained unicode artifacts from pre-fix downloads. The test score improvement (76% → 84%) shows the 100M learned more despite identical val loss.

**The most noticeable practical difference is response coherence.** The 24M degrades to plausible-sounding nonsense on topics outside its confident knowledge — it will confidently produce grammatically correct sentences with wrong or invented content. The 100M, trained on 7.8B tokens vs 2B and with a larger SFT dataset, maintains coherent reasoning even when it doesn't know the answer. As long as the user input is clean (no garbled text, no extreme prompt injection), the 100M will not produce word salad — it either gives a reasonable answer, hedges, or stays on topic. Out-of-domain queries degrade gracefully to vague-but-coherent responses rather than collapsing entirely.

---

## SFT Dataset Design (8 categories)

The 100M uses an 8-category SFT design versus the 24M's 3-category design.

| Category       | Target % | Token          | Source        |
| -------------- | -------- | -------------- | ------------- |
| greeting       | 12%      | `<sana_salute>`| HC only       |
| goodbye        | 8%       | `<sana_salute>`| HC only       |
| identity       | 10%      | `<sana_smug>`  | HC + API      |
| smalltalk      | 10%      | `<sana_smug>`  | HC + API      |
| factual        | 35%      | `<sana_think>` | HC + API      |
| empathy_sad    | 15%      | `<sana_sad>`   | HC + API      |
| empathy_happy  | 8%       | `<sana_happy>` | HC + API      |
| entity         | 2%       | any            | HC only       |

Greeting and goodbye are HC-only because API generation for these categories collapses under deduplication — `"user sends 'hi'"` causes GPT to always output `user: "hi"`, leaving one unique example per seed. The hardcoded set covers all variants.

---

## Full Pipeline

### 0. Install dependencies

```bash
pip install torch pyyaml datasets numpy tqdm openai requests
```

### 1. Download pretraining data

```bash
python pretrain/download_data.py \
    --output_dir data/pretrain_raw \
    --max_tokens 4500
```

- Target: ~4.5B raw tokens → ~7.8B BPE tokens after tokenization
- Runtime: ~30 min
- Unicode normalization applied at write time

### 2. Train BPE tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --data_dir data/pretrain_raw \
    --vocab_size 16000 \
    --sample_lines 100000 \
    --output tokenizer/sana_tokenizer/tokenizer.json \
    --force
```

**Critical:** do NOT use `--dialogue_dir`. That flag corrupts BPE merges and causes subword fragmentation ("ing vironment" instead of "environment").

### 3. Tokenize pretraining data

```bash
python pretrain/tokenize_data.py \
    --data_dir data/pretrain_raw \
    --output_dir data/pretrain_bin \
    --shard_size 200 \
    --workers 32
```

### 4. Pretrain

**H100 (80GB) — recommended:**
```bash
python pretrain/pretrain.py --config pretrain/config_100m_h100.yaml
```
- Runtime: ~3.5–4h
- Expect ~119,630 steps, val_loss ~2.05, test score 72–84%
- Delete old checkpoints every ~20k steps to avoid disk-full crash:
  ```bash
  ls checkpoints/pretrain/ckpt_*.pt | sort | head -n -3 | xargs rm -f
  ```

**Resume from checkpoint after crash:**
```bash
python pretrain/pretrain.py \
    --config pretrain/config_100m_h100.yaml \
    --resume checkpoints/pretrain/ckpt_XXXXXX.pt
```

**RTX 4060 (8GB) — not recommended for 100M:**
Training a 100M model on a 4060 takes ~5 days for a full run. Use H100 or a cloud GPU.

**Expected loss curve (H100, 100M):**

| Step    | Val loss |
| ------- | -------- |
| 16,000  | ~2.286   |
| 30,000  | ~2.217   |
| 62,000  | ~2.133   |
| 119,630 | ~2.053   |

### 5. Test pretrained model

```bash
python test_pretrain.py --checkpoint checkpoints/pretrain
```

Pass threshold: 60%. The 100M scores 72–84% depending on temperature sampling variance.

### 6. Generate SFT data

```bash
export OPENAI_API_KEY="sk-..."

python pretrain/generate_direct_sft.py \
    --output data/sft_direct.jsonl \
    --repeat 3 \
    --n_api 15000 \
    --target 15000 \
    --workers 30
```

- Runtime: ~12 min, ~$1.50 in API costs
- Expected output: ~9,000–10,000 unique examples after light deduplication
- Greeting and goodbye categories are HC-only (API generation deduplicates to near-zero)

**Verify distribution:**
```bash
python3 -c "
import json
from collections import Counter
data = [json.loads(l) for l in open('data/sft_direct.jsonl') if l.strip()]
cats = Counter(d['_meta']['category'] for d in data)
toks = Counter()
for d in data:
    for t in d['conversations']:
        if t['role'] == 'sana':
            c = t['content']
            for tok in ['<sana_salute>','<sana_happy>','<sana_think>','<sana_sad>','<sana_smug>']:
                if c.startswith(tok): toks[tok] += 1; break
print(f'Total: {len(data)}')
print('Categories:', dict(sorted(cats.items())))
total_t = sum(toks.values())
[print(f'  {t:<18} {c:>4}  ({c/total_t*100:.0f}%)') for t,c in sorted(toks.items(), key=lambda x:-x[1])]
"
```

Target token distribution: smug ~25%, sad ~22%, think ~20%, happy ~15%, salute ~14%.

### 7. Fine-tune

Update `pretrain_checkpoint` in `finetune/config_100m.yaml` to point at your best pretrain checkpoint, then:

```bash
python finetune/finetune.py --config finetune/config_100m.yaml
```

- Runtime: ~45–60 min on RTX 4060
- 7 epochs saved to `checkpoints/finetune/`
- **Use epoch 3** — sweet spot for this dataset size

**Note on reported avg loss:** `finetune.py` reports avg_loss at ~4× the real value due to gradient accumulation interaction. Real epoch loss = reported ÷ 4. Real sweet spot: 0.45–0.55 real avg loss (reported: ~1.8–2.2).

### 8. Test fine-tuned model

```bash
# Test the sweet spot checkpoint
python test_finetune.py
```

Pass threshold: 70%. The 100M consistently reaches 90.9% at epoch 3.

### 9. Interactive inference

```bash
python inference/inference.py \
    --checkpoint checkpoints/finetune \
    --temperature 0.4
```

**Important:** set `max_history_turns=1` in `inference.py` before running. The SFT training data is mostly single-turn — 3 turns of history causes context bleeding from previous exchanges.

```python
# inference/inference.py — SanaInference.__init__
max_history_turns: int = 1,   # change from 3 to 1
```

Commands at the prompt:
- `reset` — clear conversation history
- `quit` / `exit` — exit

---

## Config Files

### `pretrain/config_100m_h100.yaml`
```yaml
model:
  hidden_dim: 640
  num_layers: 18
  num_heads: 10
  ffn_multiplier: 2.667
  vocab_size: 16000
  max_seq_len: 512

training:
  batch_size: 128          # lowered from 256 due to VRAM
  learning_rate: 2.0e-4
  warmup_steps: 500
  bf16: true
  use_compile: true
```

Batch size was lowered from 256 to 128 during the actual run due to VRAM constraints. This automatically doubled the step count to maintain the same token coverage.

### `finetune/config_100m.yaml`
```yaml
training:
  pretrain_checkpoint: checkpoints/pretrain/ckpt_119630.pt
  learning_rate: 1.0e-5
  num_epochs: 7
  batch_size: 4
  gradient_accumulation: 4   # effective batch = 16
  warmup_steps: 200
```

---

## Repo Structure

Identical to the 24M edition plus the 100M-specific configs:

```
sana/
├── model/
│   └── model.py
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── tokenizer.py
│   └── sana_tokenizer/
│       └── tokenizer.json
├── pretrain/
│   ├── download_data.py
│   ├── tokenize_data.py
│   ├── pretrain.py
│   ├── generate_direct_sft.py   ← 8-category design for 100M
│   ├── config.yaml              ← 24M RTX 4060
│   ├── config_h100.yaml         ← 24M H100
│   ├── config_100m_h100.yaml    ← 100M H100  ← new
│   └── config_100m_4060.yaml    ← 100M RTX 4060 (not recommended)
├── finetune/
│   ├── finetune.py
│   ├── config.yaml              ← 24M SFT config
│   └── config_100m.yaml         ← 100M SFT config  ← new
├── inference/
│   └── inference.py
├── test_pretrain.py
├── test_finetune.py
├── README.md                    ← 24M edition
└── README_100M.md               ← this file
```

---

## Key Differences from 24M

**More SFT data is required.** The 24M worked at ~1,200 examples × 5 epochs = 375 optimizer steps. The 100M has 4× more parameters and needs proportionally more SFT signal. This run used ~9,700 examples × 7 epochs = ~4,200 optimizer steps. The sweet spot is epoch 3 (~1,800 steps) — the model learns the behavioral format cleanly before starting to overfit.

**SFT teaches format, not facts (same principle, more important at scale).** At 100M the pretrain knowledge is richer and harder to override. Heavy SFT (more epochs, more repetition) degrades factual quality more than at 24M. Keep epoch count conservative and use the earliest checkpoint that passes the behavioral tests.

**Greeting/goodbye require HC-only training.** Generic API seeds for greetings collapse under deduplication — GPT generates `user: "hi"` for any seed phrased as "user sends hi", leaving one unique example per seed. The HC set covers all variants (`hi`, `hello`, `hey`, `sup`, `yo`, `good morning`, etc.) with repeat=3 × 5 = 15 copies each, which is sufficient.

**Token-level loss imbalance.** Short responses (greetings: 1-3 tokens) have much less gradient signal than long responses (factual: 15-30 tokens). The greeting emotion token (`<sana_salute>`) is systematically under-trained relative to `<sana_think>`. This causes occasional greeting token misfires (✨ instead of 🪼) that are cosmetic — the content routing is correct. A first-token weighted loss in `finetune.py` would fix this in a future iteration.
