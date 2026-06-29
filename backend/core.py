"""
core.py — THE CONTRACT between the frontend (PySide6 UI) and the backend (the model engine).

This file is the single seam between the two halves of the app. Nothing else crosses the
line: the UI only ever *calls the methods* and *connects to the signals* declared here, and
the backend only ever *fills in the method bodies* and *emits the signals* declared here.

Why this design works (and why there is no WebSocket / no IPC / no server):
    We chose PySide6, so the UI and the model live in the SAME Python process. That means
    "how the two halves talk" is just ordinary in-process Python:

        FE -> BE   is a plain method call.        terminal types `g`  ->  engine.continue_()
        BE -> FE   is a Qt signal.                hook fires          ->  engine.paused.emit(...)

    The only subtlety is THREADS. The model's forward pass is slow and BLOCKS, so it must run
    on a worker QThread — never on the GUI thread, or the window freezes solid. Qt makes the
    cross-thread direction safe for us: a signal emitted on the worker thread is delivered to a
    slot on the GUI thread via a *queued connection* automatically. So the backend can `emit`
    from inside a PyTorch hook on the worker thread, and the UI slot runs safely on the GUI
    thread. We never have to write a single lock or queue for the BE->FE direction.

    The FE->BE direction (method calls) runs on the GUI thread. Most methods just flip a flag
    or set a threading.Event that the worker thread is waiting on — they must return *instantly*
    and must NOT do model work themselves, or, again, the window freezes.

Thread rule, stated once so every comment below can assume it:
    * GUI thread:    builds widgets, calls Engine methods, receives signals. Owns all QWidgets.
    * Worker thread: runs the model, emits signals, blocks on the resume Event at breakpoints.
    * The worker thread must NEVER touch a QWidget. It talks to the UI only by emitting signals.

This file imports and runs on its own (the methods raise NotImplementedError). The backend
owner fills the bodies; the frontend owner builds widgets against the signals. Either side can
develop in parallel — see fake_engine.py, which subclasses this and emits canned signals so the
whole UI can be built and tested with no model loaded at all.
"""

from PySide6.QtCore import QObject, Signal


class Engine(QObject):
    """The debugger backend, viewed as a contract.

    Construct ONE of these on the GUI thread and hand it to the UI. The UI connects every signal
    below to a slot, then drives the engine by calling the methods below in response to user
    actions (picking a model, typing in the terminal, clicking a breakpoint).
    """

    # =================================================================================
    # SIGNALS — the BE -> FE direction.
    #
    # Every signal here is emitted BY THE BACKEND (often from the worker thread, inside a
    # PyTorch forward hook) and received BY THE FRONTEND on the GUI thread. The frontend
    # connects each one in its constructor, e.g.  engine.paused.connect(self.on_paused).
    #
    # Keep every payload SMALL. Because UI and model share a process, you *can* shove a raw
    # multi-megabyte tensor through a signal — but then the GUI thread chokes trying to render
    # it. The backend must reduce/aggregate before emitting (see reduce.py). The only place a
    # large-ish value is allowed is `value`, and only because the user explicitly asked for it.
    # =================================================================================

    models_listed = Signal(list)
    """Emitted after list_models(directory) finishes scanning.
    Payload: list[{"name": str, "path": str}] — display name + full path for each model found.
    A valid model directory is any folder that contains a config.json (AutoConfig can load it).
    FE use: fill the startup combo box; the "path" is passed straight to load_model() and then
    to AutoConfig.from_pretrained(path), so it must be the actual directory, not just the name.
    Thread: disk scan can run inline for small dirs, or on a short worker; FE slot is GUI-thread."""

    load_progress = Signal(float)
    """Emitted repeatedly while load_model() pulls weights into memory. Payload: 0.0..1.0.
    FE use: drive a progress bar on the startup page. Loading a big model takes seconds, so this
    is the user's only feedback that anything is happening. Emitted from the worker thread."""

    ready = Signal(dict)
    """Emitted once the model + tokenizer are fully loaded and the engine can accept run().

    Payload — all fields come directly from AutoConfig.from_pretrained(path):
        layers      int   config.num_hidden_layers
        hidden      int   config.hidden_size
        heads       int   config.num_attention_heads       — query heads
        kv_heads    int   config.num_key_value_heads       — key/value heads; equals `heads`
                          on standard MHA (GPT-2, early Llama), LESS than `heads` on GQA
                          models (Llama-2 70B: heads=64, kv_heads=8). The UI shows this
                          separately because it changes what the attention slice means.
        mlp_size    int   config.intermediate_size         — the width of the FFN/MLP block.
                          GPT-2: 4×hidden. Llama SwiGLU: ~8/3×hidden (so ~11008 for 7B).
                          This is a DIFFERENT population of neurons from the attention hidden
                          units — not the same thing as `hidden`.
        model_type  str   config.model_type               — "gpt2", "llama", "mistral", etc.
                          The backend needs this to know which submodule names to hook:
                            gpt2/falcon  → model.transformer.h[i]
                            llama/mistral/phi → model.model.layers[i]
                          The frontend uses it only for display labels.
        vocab_size  int   config.vocab_size
        max_seq     int   config.max_position_embeddings

    FE use: switch the startup QStackedWidget to the debugger page, populate the misc dock
    and the layer rail (every row's label uses hidden/heads/kv_heads/mlp_size).
    Until this fires, run()/set_breakpoint() are not valid. Emitted from the worker thread."""

    status = Signal(str)
    """Emitted continuously during a run to narrate progress, e.g. "tokenizing…",
    "layer 5/12", "generating token 8". Payload: a short human string. FE use: the top strip of
    the central pane (the 'track its progress' bar from the design). This is cheap, fire it
    liberally. Emitted from the worker thread."""

    tokenized = Signal(list)
    """Emitted once right after the prompt is turned into tokens, before the forward pass.
    Payload: list[str] of token texts (already decoded for display). FE use: render the input
    token stream at the top of the display so the user sees how their prompt was split. The
    design calls out a breakpoint right here — the engine may pause immediately after emitting
    this (see paused). Emitted from the worker thread."""

    paused = Signal(dict)
    """THE CORE DEBUGGER EVENT. Emitted when execution hits a breakpoint (or a break-in) inside
    the forward pass, *immediately before the worker thread blocks* waiting for continue_().
    Payload: {"location": str, "snapshot": <small reduced dict>} — where we stopped (e.g.
    "block.5") and a small summary of activations at that point (norms, top-k neurons,
    downsampled attention — NOT raw tensors).
    FE use: update the 'current location' in the misc dock, print a line in the terminal
    ("Breakpoint hit at block.5"), and feed snapshot to the viz widget.
    Interaction flow: after this fires, the worker thread is BLOCKED on a threading.Event. The
    UI is fully responsive (it's the GUI thread, untouched). Nothing advances until the user
    types `g` and the FE calls continue_(), which sets that Event. Emitted from the worker."""

    activations = Signal(dict)
    """Emitted as the forward pass flows through layers WITHOUT stopping (the live 'neurons
    firing' view). Payload: {"layer": int, "summary": <small reduced dict>}. FE use: animate the
    viz widget. Difference from `paused`: this does NOT stop execution — it's a fire-and-forget
    progress frame. The backend MUST throttle these (every layer × every token is a flood); cap
    the rate or only emit on a subset of layers, and the FE should tolerate dropped frames.
    Emitted from the worker thread."""

    output = Signal(str)
    """Emitted once per generated token as decoding proceeds. Payload: the new token's text.
    FE use: append to the output stream in the central pane (this is the model's answer
    appearing live). Emitted from the worker thread."""

    finished = Signal()
    """Emitted when a run completes (generation stopped / EOS / max tokens). No payload.
    FE use: re-enable the prompt input, mark the terminal idle, clear 'running' state. Exactly
    one finished follows each successful run(). Emitted from the worker thread."""

    value = Signal(str, object)
    """The reply to an inspect() request. Payload: (ref, data) — the thing asked for and its
    value. This is the ONE signal allowed to carry a larger payload, because the user explicitly
    asked to see a specific slice. Even so, inspect() should return a *slice*, not a whole layer.
    FE use: print the value in the terminal scrollback. Emitted from the worker thread (the
    inspect runs against the paused state living on that thread)."""

    error = Signal(str)
    """Emitted on any failure — model load failed, bad command, exception in the forward pass.
    Payload: a human-readable message. FE use: print it red in the terminal and unwind any
    'running'/'loading' UI state. Every method below may answer with this instead of its normal
    signal. Emitted from whichever thread hit the error."""

    # =================================================================================
    # METHODS — the FE -> BE direction.
    #
    # Every method here is CALLED BY THE FRONTEND on the GUI thread, in response to a user
    # action. The backend implements them. A method either:
    #   (a) kicks off work on the worker thread and returns immediately (load_model, run), or
    #   (b) pokes a flag / threading.Event that the worker thread is watching (continue_,
    #       step, break_in, set_breakpoint), and returns immediately.
    # NOTHING here may block the GUI thread on model work. The "answer" always comes back later
    # as one of the signals above, never as a return value (except trivial sync queries).
    # =================================================================================

    def list_models(self, directory: str) -> None:
        """Called when the user clicks Scan on the startup page. Walk `directory` looking for
        subdirectories that contain a config.json (the minimal requirement for
        AutoConfig.from_pretrained to succeed). Emit models_listed with the results.
        Returns immediately — do the scan inline (it's fast) or on a short worker thread.
        Answer via models_listed; failure via error. Never blocks the GUI."""
        raise NotImplementedError

    def load_model(self, name: str) -> None:
        """Called when the user picks a model and confirms. Start loading the model + tokenizer
        ON THE WORKER THREAD and return immediately. Progress arrives via load_progress; success
        via ready; failure via error. Do NOT load on the GUI thread — big models take seconds and
        the window would freeze."""
        raise NotImplementedError

    def run(self, prompt: str) -> None:
        """Called when the user submits a prompt. Spawn the worker thread that tokenizes, runs the
        forward pass, and generates. Returns immediately; everything else streams back as signals:
        tokenized -> (paused at breakpoints) -> activations/output … -> finished.
        Only valid after ready has fired. A second run() should be rejected (via error) while one
        is in flight."""
        raise NotImplementedError

    def set_breakpoint(self, target: str) -> None:
        """Called when the terminal parses `bp <target>` (e.g. "block.5", "block.5.attn").
        Record the breakpoint so the matching forward hook will pause next time it fires. This is
        just bookkeeping on a shared set — it returns instantly and is safe to call mid-run (the
        worker thread reads the set inside each hook). Confirm via status or a terminal line."""
        raise NotImplementedError

    def clear_breakpoint(self, target: str) -> None:
        """Called for `bc <target>` / clicking a breakpoint off. Remove it from the shared set.
        Same instant, lock-light bookkeeping as set_breakpoint."""
        raise NotImplementedError

    def continue_(self) -> None:
        """Called when the user types `g` (go) while paused at a breakpoint. Sets the resume
        threading.Event the worker thread is blocked on, so the forward pass proceeds to the next
        breakpoint (or to completion). Trailing underscore because `continue` is a Python keyword.
        threading.Event.set() is thread-safe, so the GUI thread calls it directly. Returns
        instantly; the next thing the FE sees is another paused, more output, or finished."""
        raise NotImplementedError

    def step(self) -> None:
        """Called for `p`/`t` (single-step): like continue_, but arm a one-shot breakpoint on the
        very next hook so execution advances exactly one layer and pauses again. Implemented as
        'set resume Event, but also set a step flag every hook checks'. Returns instantly;
        answers with the next paused."""
        raise NotImplementedError

    def inspect(self, ref: str) -> None:
        """Called for `inspect <ref>` while paused (or running). Read a *small slice* of the
        current state — a tensor by name, a weight, a neuron's value — and answer via the value
        signal. Runs against the paused snapshot on the worker thread. Keep the returned slice
        small; do not stream a whole layer just because the user named it."""
        raise NotImplementedError

    def break_in(self) -> None:
        """Called for `break` / Ctrl-Break while a run is flowing freely (no breakpoint set, or
        between them). Sets a 'pause as soon as possible' flag that every forward hook checks on
        entry, so the next hook to fire pauses and emits paused. This is the WinDBG 'break in at
        any time' behaviour. Returns instantly; the pause itself happens on the worker thread at
        the next hook boundary."""
        raise NotImplementedError
