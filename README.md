# Sayla

A personal, from-scratch LLM that can cover math, physics, and coding (C, C++, and python)
— tokenizer, embeddings, attention, and a hand-written neural network, all in plain C — wrapped
in a small raylib desktop app with a chat interface, a live voice mode, and a hand-drawn bird avatar. 
There's no PyTorch, no TensorFlow, no external ML runtime underneath it: every matrix multiply 
and every gradient in this project is code you can read start to finish. Speech-to-text is the 
one place an external model is involved (whisper.cpp, optional); text-to-speech is a from-scratch 
formant synthesizer with no external library or model file at all.

This documents what the LLM can actually do and what it can't do.

<img width="1072" height="744" alt="Screenshot 2026-08-13 103426" src="https://github.com/user-attachments/assets/e20d8c8e-15b3-4359-b5b7-87aa8331fe99" />
<img width="1080" height="744" alt="Screenshot 2026-08-13 103335" src="https://github.com/user-attachments/assets/79e4c9d2-7196-4c28-9219-9283a23f190f" />

---

## What Sayla can do

- **Chat.** Type a message, get a reply. Message bubbles scroll, wrap
  correctly (including around embedded newlines), and can be clicked to copy
  their text. The sidebar keeps a list of past prompts.
- **Domain-specialized answers.** Every prompt is routed to one of three
  independently trained models — **math**, **physics**, or **coding** — based
  on keyword matching, with typo tolerance (a misspelled "derivative" still
  routes to math). A prompt that matches nothing gets an "I'm not sure
  about that" reply instead of being run through the coding model
  anyway — see **Domain routing and honesty** below.
- **Multiple model architectures, switchable without losing the old one.**
  Sayla isn't tied to one fixed model shape. Each registered model (see
  `ModelConfig` in `engine.h`) carries its own context length, embedding
  dimension, attention head size, hidden layer size, epoch count, and
  learning rate, with its own namespaced cache files, so a new architecture
  can be added and trained without touching or retraining an existing one.
  The sidebar's model picker (bottom-left) shows the current model, lets you
  browse all registered ones, and switches between them. **Oriole 1** is the
  only model registered currently; adding another is a single entry in
  `MODEL_REGISTRY[]`.
- **Adjustable effort.** The same picker also sets effort (Low/Standard/
  High), which genuinely changes how a reply is generated, not just a label:
  higher effort runs multiple independent generation attempts and keeps
  whichever one scores as least garbled (see `scoreResponseQuality()` in
  `engine.c`), since generation is already stochastic. Low effort caps
  token count lower for a faster, cheaper attempt. This is a real,
  externally-measurable tradeoff — high effort takes roughly 3x as long as
  standard for a three-attempt request, not a cosmetic setting.
- **File attachments.** Drag and drop text files onto the input box; their
  content is embedded into your message (with a per-file token count shown as
  a chip) so the model actually sees the file, not just its name.
- **A Live voice panel.** Click the mic once to start listening —
  it's a toggle, not push-to-talk. A from-scratch voice activity detector
  (VAD) decides when you're actually speaking, so there's no button to hold.
  Your speech is transcribed, sent through the same reply pipeline as typed
  text, and the reply is spoken back — through a formant synthesizer built
  from scratch (resonant filters, a hand-built phoneme table, spelling-to-
  sound rules), not an external TTS library or model file, so it works even
  in builds with voice *listening* turned off entirely. An audio-reactive
  ring around the avatar shows real playback/capture amplitude, not a canned
  animation. When the last reply came from a genuine coding-domain match,
  Sayla says so aloud and shows a code panel instead of reading source code
  character by character. The panel supports multiple named files with
  per-file tabs and a working copy-to-clipboard button, though the model
  itself doesn't yet produce multi-file-formatted output (see
  **Limitations**).
- **A live training screen.** The first run (or any run after a corpus file
  changes, or after switching to a model architecture that's never been
  trained) trains the affected domain models from scratch, with a real-time
  progress screen — current step, total steps, running loss — instead of a
  frozen window. Every run after that loads from a cache in well under a
  second, until a corpus or architecture actually changes.

## Domain routing and honesty

A prompt that doesn't match any domain's keywords doesn't fall
through to the coding model and get whatever garbled text that model
happens to produce — `generateResponse()` in `engine.c` checks this first
and returns a short, hand-written reply instead ("I'm not sure
about that one — try asking me about math, physics, or code"), rotating
through a few variants. A separate, similar check catches greetings
("hi", "hey", "how are you") and replies in kind, also with real variety.
Neither of these is model-generated text — this project has been explicit
throughout that the underlying attention model (see **Limitations**) can't
reliably produce coherent prose like this on demand, so these specific,
common interaction patterns get a reliable, hand-written answer instead of
an attempt from a model that would likely fail at it. Generated text that
*is* model output also gets a pass to strip out a real, observed failure
mode: with a short context window, the model can drift into hallucinating
a fake "User:"/"Assistant:" continuation partway through a reply — the
tail of that hallucination is cut before display, not shown.

## What it's built on

### The model itself

- **Byte-level BPE tokenizer** (`BPE.c`/`BPE.h`) — starts from the 256 raw
  byte values and learns merge rules from the training corpus (up to 1000
  vocabulary entries per domain by default). Pair counting and merge-rank
  lookups both go through an open-addressing hash table rather than a linear
  scan — an earlier linear-scan version effectively never finished tokenizing
  anything past a few hundred KB of text; this was a real, since-fixed
  performance bug, not a design choice.
- **Token + positional embeddings** (`embedding.c`/`embedding.h`) — a learned
  vector per token (128-dimensional for Oriole 1), plus a matching
  positional embedding per slot in the context window, added together
  before attention. Dimension is a per-model value now, not a fixed
  constant — see **Multiple model architectures** above.
- **Single-head causal self-attention** (`attention.c`/`attention.h`) — real
  Q/K/V projections, scaled dot-product scores, causal masking (a position
  can only attend to itself and earlier positions), softmax, and a fully
  hand-derived backward pass. Verified independently on a synthetic
  "copy-the-token-from-N-positions-back" task before being wired into the
  model, specifically because that task is unsolvable without attention
  actually working. Sized entirely from constructor arguments
  (`initAttentionLayer`), not hardcoded constants, which is what makes
  registering a genuinely different-shaped model possible without touching
  this file at all.
- **A small feed-forward network** (`neuron.c`, `layer.c`, `network.c`) — one
  sigmoid hidden layer (512 neurons for Oriole 1), then a real softmax +
  cross-entropy output layer over the vocabulary, trained with plain
  stochastic gradient descent. Like the attention layer, shaped entirely by
  constructor arguments (`initNetwork`'s `layerSizes[]`), not fixed
  constants.
- **32 tokens of context** per prediction, for Oriole 1 — the model looks at
  the last 32 tokens, runs them through attention, and predicts the next
  one. This is the single biggest constraint on output quality (see
  **Limitations** below), and it's a per-model value now: a differently
  registered model could use a different context length entirely, at a
  real compute cost (the attention score matrix is quadratic in context
  length, not linear, so this doesn't come free). Measured directly going
  from 16 to 32 tokens of context: roughly 1.7x slower per training step —
  meaningfully less than a naive quadratic-scaling estimate (~4x) would
  suggest, since other parts of the training pipeline don't scale with
  context length at all and dilute that cost in practice.

### Three domain models, not one

Rather than one shared vocabulary trying to cover math notation, physics
terminology, and three programming languages at once, Sayla trains three
separate tokenizer + embedding + attention + network stacks — **math**,
**physics**, **coding** — each on its own specialized corpus, concatenated
with a shared conversational base corpus so every domain can still handle
small talk and "what is X" questions without diluting its specialty
vocabulary. The domain-specific corpus text is repeated enough times in
that concatenation to avoid being outweighed by the (often much larger)
shared base corpus — a real, measured problem for the smaller math/physics
corpora specifically, not a hypothetical one.

At response time, a keyword classifier (with Levenshtein-distance typo
tolerance layered on top of exact substring matching) picks which domain
model actually answers. This is a coarse heuristic, not a learned router.
Short, substring-risky keywords (like "hi" or "yo") are matched by exact
word boundary rather than substring containment specifically because the
substring version produced real false positives during testing (e.g. "yo"
matching inside "your"). Unmatched or tied prompts get the honest fallback
described above, not a silent default to the coding model.

### The engine

Training and inference for all three domains, across whichever model
architecture is currently active, live in `engine.c`, along with:

- **Model caching** (`cache.c`/`cache.h`) — the trained tokenizer,
  embeddings, attention weights, and network for each domain are serialized
  to disk, keyed against the *exact byte content* of the corpus that trained
  them, plus the active model's context length and hidden layer size. Edit
  a corpus file, switch to a different registered model, or change an
  existing model's shape, and the cache is automatically invalidated on
  next launch — no manual cache-clearing step, and no risk of silently
  loading a cache file shaped for a different architecture than the one
  currently active. The on-disk format uses explicit little-endian
  serialization for every number it stores, rather than dumping the host
  machine's raw in-memory representation, so cache files are portable
  across machines regardless of endianness (verified byte-for-byte
  identical to the previous format on little-endian hosts, which covers
  essentially all mainstream hardware this targets — Windows, macOS
  Intel/Apple Silicon, and Linux x86/ARM are all little-endian).
- **Background training** — training runs on its own thread (a small
  cross-platform shim: POSIX pthreads on Linux/macOS, `SRWLOCK` + native
  Win32 threads on Windows) so the window stays responsive and can show live
  progress instead of freezing. Depending on corpus size, first-run training
  can take anywhere from seconds to hours; this is a real, unavoidable cost
  of training attention-based models from scratch on CPU, not something the
  caching layer can hide on the very first run. Switching to a model
  architecture that's never been trained pays this same cost, and currently
  does so on the calling thread (see **Limitations**).

### Voice

- **Speech-to-text:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
  running a downloaded ggml model (e.g. `ggml-base.en.bin`) fully offline.
  The only piece of voice still gated behind the `SAYLA_ENABLE_VOICE` CMake
  flag, and the only external ML dependency in the whole project.
- **Text-to-speech:** built entirely from scratch (`src/tts/`) — a
  two-pole resonant filter (`resonator.c`), a hand-built phoneme table with
  formant values from standard acoustic-phonetics references
  (`phonemes.c`), spelling-to-sound rules (`text_to_phonemes.c`), and a
  formant synthesis engine that drives three of those resonators in
  parallel from a pulsed or noise source depending on whether a phoneme is
  voiced (`format_synth.c`). No external TTS library, no model file, and
  no `SAYLA_ENABLE_VOICE` gate — speaking works in every build, since
  nothing about it depends on whisper.cpp or any other optional piece.
  Expect a clearly robotic voice; that's an inherent property of formant
  synthesis at this level of simplicity, not a bug.
- **Voice activity detection** (`src/vad/vad.c`) — also from scratch: an
  adaptive energy-threshold detector that calibrates its noise floor from
  real incoming audio for the first ~300ms after listening starts, then
  watches for sustained energy above that floor to mark speech start/end.
  This is what makes the mic a plain on/off toggle rather than
  press-and-hold — VAD decides when you're actually talking within
  whatever window is open.
- **Microphone capture:**
  [miniaudio](https://github.com/mackron/miniaudio), used directly, since
  raylib's own public API only exposes audio *playback*, not capture. Not a
  separate dependency to fetch — raylib already bundles its own copy
  internally for its audio backend, and voice capture points at that same
  bundled copy rather than fetching and compiling a second one (which
  previously caused duplicate-symbol linker errors).
- Listening (whisper.cpp, microphone capture, VAD) is gated behind
  `SAYLA_ENABLE_VOICE` and safely no-ops when it's off (Set to on by defualt).

  <img width="1071" height="745" alt="Screenshot 2026-08-13 110122" src="https://github.com/user-attachments/assets/455db1ec-11b2-4ae6-942f-aea14b25fdf8" />

### The interface

Built entirely with [raylib](https://www.raylib.com/) 5.5:

| File | Role |
|---|---|
| `window.c` | Main loop, panel switching, the training screen |
| `sidebar.c` | Chat/Live tabs, chat history list, the model + effort picker |
| `chat_view.c` | Message bubbles, scrolling, click-to-copy |
| `input.c` | Text entry, attachments, drag-and-drop |
| `live_view.c` | The Live panel: avatar, audio ring, hands-free mic toggle, multi-file code panel with copy |
| `thinking.c` | The animated "thinking" indicator shared by chat replies and the training screen |
| `avatar.c` | The hand-drawn bird avatar |
| `welcome.c`, `colors.h` | Small supporting pieces |

## Project structure

```
.
├── CMakeLists.txt
├── corpus_base.txt          # shared conversational corpus
├── corpus_math.txt          # math domain corpus
├── corpus_physics.txt       # physics domain corpus
├── corpus_coding.txt        # coding domain corpus
├── models/
│   └── ggml-base.en.bin     # whisper model 
└── src/
    ├── llm/                 # backend: tokenizer, network, engine
    │   ├── BPE.c / .h
    │   ├── embedding.c / .h
    │   ├── attention.c / .h
    │   ├── neuron.c / .h
    │   ├── layer.c / .h
    │   ├── network.c / .h
    │   ├── cache.c / .h
    │   ├── chat_history.c / .h
    │   ├── engine.c / .h
    │   └── input.c / .h
    ├── vad/                  # listening: voice activity detection, mic capture, whisper
    │   ├── vad.c / .h
    │   ├── voice.c / .h
    │   ├── audio_features.c / .h   # not yet wired into voice.c -- see Limitations
    │   └── dtw.c / .h              # not yet wired into voice.c -- see Limitations
    ├── tts/                  # speaking: from-scratch formant synthesis
    │   ├── resonator.c / .h
    │   ├── phonemes.c / .h
    │   ├── text_to_phonemes.c / .h
    │   └── format_synth.c / .h
    └── renderer/             # raylib UI
        ├── window.c / .h
        ├── sidebar.c / .h
        ├── chat_view.c / .h
        ├── live_view.c / .h
        ├── thinking.c / .h
        ├── avatar.c / .h
        ├── welcome.c / .h
        └── colors.h
```

Model caches (namespaced per registered model — e.g. `model_cache_math.bin`
for Oriole 1) and a saved chat transcript (`chat_history.txt`, written on
Ctrl+S) appear alongside the corpus files once you've run the app.

## Libraries

Every third-party library this project depends on, and what it's used for:

| Library | Used for | Required? | Link |
|---|---|---|---|
| raylib 5.5 | Windowing, rendering, input, audio playback | Always | https://github.com/raysan5/raylib |
| whisper.cpp | Speech-to-text (listening) | Only if `SAYLA_ENABLE_VOICE` is on | https://github.com/ggml-org/whisper.cpp |
| miniaudio | Microphone capture | Only if `SAYLA_ENABLE_VOICE` is on (bundled via raylib, not fetched separately) | https://github.com/mackron/miniaudio |

Everything else — the tokenizer, embeddings, attention, the feed-forward
network, training, and text-to-speech — is original code with no external
dependency. espeak-ng was used for text-to-speech earlier in this project's
history and has since been fully removed in favor of the from-scratch
formant synthesizer described above; it's not a dependency anymore.

## Dependencies

**Required:**
- [raylib](https://www.raylib.com/) 5.5 — fetched automatically via CMake
  `FetchContent`
- CMake 3.20+
- A C99 compiler (C++17 as well if voice is enabled — whisper.cpp is C++)
- pthreads (Linux/macOS) or native Win32 threading (Windows, via a small
  compatibility shim already in `engine.c`/`voice.c`)

**Optional (only if `SAYLA_ENABLE_VOICE` is on, and only for listening —
speaking has no dependencies at all):**
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) + a downloaded ggml
  model file, fetched automatically by `CMakeLists.txt`
- miniaudio — no separate fetch needed; sourced from raylib's own bundled copy


You only need to separately download a whisper model file yourself and
point `WHISPER_MODEL_PATH` (in `window.h`) at it (Make a models folder in your Release folder and put the ggml-base.en there, outside the models folder put the corpus.txt files).

<img width="628" height="268" alt="Screenshot 2026-08-13 115033" src="https://github.com/user-attachments/assets/066876ae-e324-4890-95ea-1e23b89e6e48" />

## Building

```
cd build
cmake --build . --config Release 
cd Release
./Sayla
```

The voice build pulls in whisper.cpp via `FetchContent`, which takes
noticeably longer than the base build and requires network access during
configuration. This has been built and run successfully end-to-end on real
hardware, not just checked for compile correctness — listening and
transcription are confirmed working from real console output showing
correct transcribed text. Spoken replies produce real audio too, though a
genuine buffer-starvation bug (audio breaking up during longer replies) was
found and fixed along the way; that fix hasn't been re-confirmed against
real hardware since.

## Limitations

**Model capability.** Oriole 1 is a single-head, 32-token-context attention
model — nowhere near the scale of a real production LLM. It can reliably
continue short, locally-familiar patterns it's seen often in training (to
the point of reproducing memorized phrases verbatim), but it has no
architectural mechanism for staying coherent beyond its context window.
Expect fluent-looking fragments that don't necessarily connect to each other
in a longer response — that's the model's real ceiling, not a bug to chase.
"Explaining code" is, at best, a learned association between code shapes and
comment shapes from training data, not genuine program comprehension. The
architecture now supports registering a larger model alongside
this one (see **Multiple model architectures**), but a bigger context
window or embedding dimension costs real, non-linear compute — it isn't a
free quality dial.

**The effort-level quality heuristic is a real but imperfect proxy.**
Higher effort picks the least-garbled of several generated candidates by
checking whether each word looks alphabetically plausible, which reliably
catches obvious garbling (stray digits or punctuation mid-word) but cannot
tell a real word from an alphabetic-but-meaningless fragment the model
happened to produce — it's directionally correct, not a true coherence or
dictionary check.

**Switching to an untrained model blocks the UI.** `setActiveModelIndex()`
runs the same load-or-train sequence startup does, but currently does so
directly on the calling thread rather than in the background — switching to
a model that's never been trained will visibly pause the interface for as
long as that training takes. Not an issue with only one model registered
today; worth wrapping in the same background-thread pattern
`startEngineInitAsync()` already uses once a second model actually exists
to switch to.

**A from-scratch command-recognition path exists but isn't wired in yet.**
`src/vad/audio_features.c` and `dtw.c` implement real, independently-tested
building blocks for a fixed-vocabulary command recognizer (filterbank
feature extraction and dynamic time warping) intended as a pure-C
alternative to whisper.cpp for listening. Neither is currently called from
`voice.c` — listening still goes through whisper.cpp today.

**Domain routing** is keyword matching, not a learned classifier. It
handles clear cases (and common typos) well, but a prompt with no
recognizable domain vocabulary gets the honest fallback reply described
above, not a real answer — it was never going to get one, since nothing in
the prompt told the classifier which domain to ask.

**Training data.** The coding corpus is real, license-clean source (the
project's own code plus original examples). The math and physics corpora are
intentionally small placeholders — enough to exercise the training pipeline,
not a substitute for real sourced data (Stack Exchange dumps, OpenStax
textbooks, etc., are a reasonable next step if you want these domains to
improve).

**No conversation memory.** Each reply is generated from the current message
alone — there's no mechanism that feeds prior turns back in as context beyond
what's visible in the on-screen history.

**Performance.** Training is single-threaded scalar C with no GPU
acceleration; a real-sized corpus can take hours on first run.

**Live panel's code panel** has real multi-file display infrastructure
(tabs, per-file copy) driven by a `// file: name.ext` marker convention in
the response text, but the training corpus doesn't currently produce output
in that format — in practice, it shows one file (the whole response) for
essentially every reply today. The panel is ready for multi-file output the
moment a response actually contains it; nothing about the underlying model
makes it understand when a task needs multiple files.

**Voice.** A plain mutex guards state shared between the audio callback
thread, the transcription thread, and the main thread — simple and correct
for short chat replies, but not what a professional low-latency audio
engine would do. VAD's noise-floor calibration takes roughly 300ms after
listening starts before it can detect anything, so speaking immediately
after clicking the mic can be missed. The from-scratch formant synthesizer
sounds clearly robotic by design, not by accident — see **Text-to-speech**
above.

**No authentication, no multi-user support, no encryption.** Chat history
saves to a plain text file. This is a personal, single-machine project.
