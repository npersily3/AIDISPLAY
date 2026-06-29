# LLM Debugger — Build Plan

A step-through debugger for transformer inference. Load a model, run a prompt, and inspect what is actually happening at each layer — attention patterns, residual stream state, logit evolution, and weight contributions. Patch weights in memory and watch behavior change.

---

## Design Principles

**Get to something real fast.** The interesting work is interpretability, not infrastructure. Every phase ends with something you can actually use to learn about the model.

**Own the execution loop.** Don't use `.generate()`. Implement the generation loop and sampling manually — you need to understand what these do, and it takes twenty lines.

**No unnecessary layers.** The backend and UI run in the same process. There is no IPC, no serialization, no network stack. A server layer gets added only if remote attach becomes a goal.

**Python lambda breakpoints over a DSL.** The tool's audience is programmers. A small eval-based condition system is more powerful than any command language you would design, and takes a day to build.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Inference + hooks | Python + HuggingFace `transformers` | Hook infrastructure (`register_forward_hook`) exists out of the box. Named tensors. All model formats supported. Get to the interesting parts immediately instead of reimplementing what already works. |
| UI | Dear ImGui (`pyimgui`) | Designed for debuggers and real-time inspectors. Immediate mode. Same process as backend — no IPC. Used by RenderDoc and every game engine debugger. Closest spiritual match to WinDbg's design philosophy. |
| Target models | Llama 3.1 8B, Mistral 7B | Small enough to run on CPU for debugging. Well-documented architectures. Representative of modern transformer design. |

### What was rejected and why

**Rust backend** — the backend is not performance-critical. You are stepping through inference manually. Rust buys nothing here and costs weeks reimplementing what HuggingFace already does correctly.

**C# / WPF frontend** — wrong language for a Python-backed tool. Every backend change requires a context switch to C#, a recompile, and type error triage. WinDbg's UI is good because of its *model* (command console primary, panels secondary), not because of WPF. Dear ImGui replicates that model and stays in the same process.

**gRPC / named pipes** — there is no IPC boundary. Backend and UI are in the same Python process. A server layer adds complexity with no benefit until remote attach is a real goal.

**CLI command language** — Python lambda predicates are more expressive than any DSL you would design, and the implementation is `eval()`. The audience is programmers.

---

## Phase 1 — Walking Skeleton

**Goal:** Load a model, run a prompt, display something real. End of day one.

**What to build:**
- Load model with `AutoModelForCausalLM`, tokenizer with `AutoTokenizer`
- Single forward pass, get logits out
- Register one `register_forward_hook` on the final layer to confirm the hook system fires
- ImGui window: model name, prompt input, top-5 predicted next tokens with probabilities

**End state:** Type a prompt, see what the model thinks comes next. Ugly, minimal, working.

**What you will learn:** ImGui immediate mode pattern. HuggingFace model loading. How to extract anything from a forward pass.

---

## Phase 2 — Generation Loop with Token Stepping

**Goal:** Step through autoregressive generation one token at a time.

**What to build:**
- Manual generation loop — tokenize, forward pass, sample next token, append, repeat
- Pause between tokens via `threading.Event`; UI releases on Step / Run
- Implement sampling from scratch: greedy, temperature, top-p. Passing flags to HuggingFace skips the understanding. The implementation is twenty lines.
- KV cache: implement it. Without it you re-run the full sequence every token. Understanding why you need it is more valuable than the speedup.

**ImGui panels:**
- Token stream — generated tokens appear one at a time in step mode
- Logit distribution — bar chart of top-10 candidates at current step
- Sampling controls — temperature and top-p sliders

**What you will learn:** KV cache mechanics. How temperature actually reshapes a distribution (more dramatic than expected). Why top-p is better than top-k.

---

## Phase 3 — Layer Stepping and Residual Stream

**Goal:** Step inside a single forward pass. Inspect state at each sublayer.

**What to build:**
- `register_forward_hook` on every sublayer: RMSNorm, attention, MLP
- Each hook captures input/output tensors and blocks on `threading.Event`
- `output_attentions=True` to extract attention weights from the attention sublayer
- Two-level stepping: Token (outer loop) and Layer (inner loop)

**ImGui panels:**
- Layer stack — all layers listed, current position highlighted, click to jump
- Residual stream bar — norm magnitude per layer. This single view shows where the model is doing work.
- Attention heatmap — per-head, seq_len × seq_len grid, viridis colormap, rendered via `draw_list` as a bitmap
- Live logit bar — top-5 predicted next tokens *at current layer depth*, updating as you step. Watch the model's opinion of the next token change across layers.

**What you will learn:** What the residual stream actually is. Not "skip connections" — the primary data structure that every layer reads from and writes to. Attention pattern differences between early layers (local, syntactic) and late layers (semantic, task-oriented). The `output_attentions=True` memory cost and why production inference never uses it.

**This is where the project pays off for the first time.**

---

## Phase 4 — Weight Editing and Ablation

**Goal:** Patch weights in memory, rerun the same prompt, observe what changes.

**What to build:**
- Patch command: `model.layers[i].self_attn.o_proj.weight.data[row] = 0` — one line per patch
- Snapshot before patch, restore on demand
- Ablation presets:
  - Zero an attention head (zero the output projection rows for head H) — isolates what that head contributes
  - Zero an MLP neuron — tests superposition hypotheses
  - Clamp an attention pattern to uniform — forces a head to ignore positional information
- Rerun same prompt with patch applied, compute per-layer residual delta ‖patched − original‖

**ImGui panels:**
- Patch list — active patches, toggle/remove
- Ablation shortcuts — "Zero head H in layer L" as a single action
- Delta overlay — original vs patched residual norms on the same bar chart. Side-by-side full tensor comparison is noise; the norm delta per layer is the signal.

**What you will learn:** Ablation as empirical method — the only way to establish that a component is doing something specific. Superposition: why zeroing one neuron sometimes has no effect and sometimes breaks a behavior entirely. You will find attention heads with interpretable functions. That is the payoff of this phase.

---

## Phase 5 — Breakpoints and Watch Expressions

**Goal:** Define conditions that pause execution automatically.

**What to build:**
- Breakpoint registry: a list of Python lambdas evaluated inside each hook
- State dict passed to each lambda: layer index, head index, residual norm, attention entropy, current top-1 token
- If the lambda returns true, block on `threading.Event`
- Watch expressions: same mechanism returning a scalar, accumulated into a time series per layer

```python
# Example breakpoints written directly into the UI input box
lambda s: s['layer'] == 12 and s['attn_entropy'][3] < 0.5
lambda s: s['residual_norm'] > 50.0
lambda s: s['top1_token'] == 'Paris'
```

**ImGui panels:**
- Breakpoint list — multiline Python input, enable/disable, delete
- Watch chart — scalar value across all layers, one line per watch expression

**Why this comes after weight editing:** Breakpoints are most useful once you know what to look for. Ablation first gives you the intuition to write non-trivial conditions.

**What you will learn:** Designing a condition system. What anomalous residual norms look like in practice. Attention entropy as a signal (low entropy = head is attending sharply to specific tokens).

---

## Phase 6 — Process Attach

**Goal:** Attach to a model running outside your tool.

**Approach — `.pth` injection:**
Drop a `.pth` file into the Python environment. It registers `register_forward_hook` calls into any `nn.Module` at import time and opens a socket back to your debug server. Works without modifying the target process. Works for any HuggingFace model.

**Approach — custom llama.cpp build:**
Fork llama.cpp, add hook callbacks at each layer boundary, expose a socket your server connects to. More work to maintain against a fast-moving codebase, but covers models running outside Python entirely.

**Not in scope:** Raw process attach via Windows `DebugActiveProcess` and CUDA driver memory reads. Genuinely interesting as a standalone systems project. Not a phase of this one.

**What you will learn:** Python import machinery and `.pth` injection. The gap between owning the inference and observing someone else's.

---

## Milestone Summary

| Phase | Deliverable | What you learn |
|---|---|---|
| 1 | Load model, run prompt, see top-k logits | ImGui, HuggingFace basics |
| 2 | Step through token generation | KV cache, sampling math |
| 3 | Step through layers, attention heatmaps, residual stream | Transformer internals at implementation depth |
| 4 | Weight editing, ablation presets, delta view | Circuit isolation, superposition |
| 5 | Python lambda breakpoints, watch charts | Attention entropy, condition design |
| 6 | Process attach via `.pth` injection | Python import machinery |

Phases 1–5 to a genuinely useful research instrument: roughly one month of focused work.