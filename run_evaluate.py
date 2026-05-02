"""
Wrapper around evaluate.py that caps AlignScore batch_size to 4 to avoid
OOM in Docker's default 7.65 GB VM (RoBERTa-large uses ~6.4 GB baseline).
"""
import alignscore.utils as _au

_orig_init = _au.AlignScorer.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("batch_size", 4)
    if kwargs.get("batch_size", 4) > 4:
        kwargs["batch_size"] = 4
    _orig_init(self, *args, **kwargs)

_au.AlignScorer.__init__ = _patched_init

# Hand off to the real evaluation script
import runpy
runpy.run_path("evaluate.py", run_name="__main__")
