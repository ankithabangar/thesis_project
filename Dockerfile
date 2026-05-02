# Use amd64 platform — required for qafacteval (spacy 2.2.4) and alignscore (torch<2)
# which are incompatible with Apple Silicon natively.
FROM --platform=linux/amd64 python:3.9-slim-bullseye

WORKDIR /app

# System deps for building C extensions (spacy, thinc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install wheel/setuptools pinned for compatibility
RUN pip install --upgrade "pip==23.3.1" "setuptools==59.8.0" wheel

# ── Core metrics (versions compatible with Python 3.9) ────────────────────────
RUN pip install \
    rouge-score==0.1.2 \
    bert-score==0.3.13 \
    "nltk>=3.8,<3.9.3"

# ── torch 1.x (required by alignscore; CPU-only for portability) ──────────────
RUN pip install torch==1.13.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── transformers pinned for summac + torch 1.13.1 + allennlp compatibility ────
# transformers>=4.41 imports LRScheduler from torch (added in torch 2.0) which
# breaks with torch 1.13.1. Pin <4.41 to keep everything working.
RUN pip install \
    "transformers>=4.35,<4.41" \
    tokenizers \
    sentencepiece \
    "protobuf==3.20.0"

# ── summac (--no-deps skips its overly-conservative huggingface-hub pin) ──────
RUN pip install summac --no-deps

# ── minicheck (not on PyPI — install from GitHub) ─────────────────────────────
RUN pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"

# ── qafacteval and its full dep chain ─────────────────────────────────────────
# The PyPI wheel for qafacteval (qafacteval-0.10) is an empty stub (3.3KB, no
# source files). Install from the actual GitHub source instead.
# qaeval hard-pins torch==1.6.0; we have 1.13.1 (API-compatible) → --no-deps.
# allennlp 1.1.0 also pins torch<1.7 → --no-deps.
# spacy 2.2.4 needs SETUPTOOLS_USE_DISTUTILS=stdlib (msvccompiler fix).
# spacy 2.2.4 has multiple Python 3.9 + Cython incompatibilities.
# spacy 2.3.7 (last 2.x release, Jan 2021) explicitly supports Python 3.9
# and still satisfies allennlp 1.1.0's spacy<3,>=2.1 requirement.
RUN SETUPTOOLS_USE_DISTUTILS=stdlib pip install spacy==2.3.7 && \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
# qafacteval's setup.py uses find_packages() which misses the top-level
# qafacteval.py flat module. Clone to /opt and install with -e so the repo
# directory itself is on sys.path (making qafacteval.py importable).
RUN git clone --depth=1 https://github.com/salesforce/QAFactEval.git /opt/QAFactEval && \
    python3 -c "p='/opt/QAFactEval/lerc_quip.py'; t=open(p).read(); open(p,'w').write(t.replace('self.device = cuda_device', 'self.device = \"cpu\" if cuda_device < 0 else cuda_device'))" && \
    pip install -e /opt/QAFactEval --no-deps && \
    pip install qaeval --no-deps && \
    pip install allennlp==1.1.0 --no-deps && \
    pip install "click==7.1.2" edlib "overrides==3.1.0" boto3 jsonpickle h5py tensorboardX

# allennlp 1.1.0: patch model.py to load state_dict with strict=False so that
# BART's tied lm_head.weight (absent from older checkpoints) doesn't cause a
# RuntimeError when using transformers>=4.18 which added it as a separate key.
RUN python3 -c "p='/usr/local/lib/python3.9/site-packages/allennlp/models/model.py'; t=open(p).read(); open(p,'w').write(t.replace('model.load_state_dict(model_state)', 'model.load_state_dict(model_state, strict=False)'))"

# qaeval/generation/model.py: patch for transformers 4.x API changes
# (generation_mode removed, past_key_values format changed, cache dict rewrite)
COPY patch_qaeval.py /tmp/patch_qaeval.py
RUN python3 /tmp/patch_qaeval.py

# ── alignscore ────────────────────────────────────────────────────────────────
# Install with --no-deps to skip the spacy>=3.4 pin.
# pytorch-lightning pulls in torch>=2, so install it --no-deps to keep torch 1.13.1.
RUN pip install "alignscore @ git+https://github.com/yuh-zha/AlignScore.git" --no-deps && \
    pip install "pytorch-lightning<2,>=1.7.7" --no-deps && \
    pip install \
        "datasets<3,>=2.3.2" \
        "jsonlines<3,>=2.0.0" \
        "numpy<2,>=1.23.1" \
        "scikit-learn<2,>=1.1.2" \
        "scipy<2,>=1.8.1" \
        "tensorboard<3,>=2.12.0" \
        "lightning_utilities" \
        "torchmetrics"

# ── NLTK data ─────────────────────────────────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# Copy project files
COPY . .

CMD ["python", "run_evaluate.py"]
