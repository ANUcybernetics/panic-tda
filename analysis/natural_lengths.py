"""Post-pilot GPU checks for decision-01 (see TASK-85).

Run with the Snex venv interpreter once the caption pilot has finished and
the GPU is free:

    _build/dev/snex/projects/Elixir.PanicTda.Models.PythonInterpreter/venv/bin/python analysis/natural_lengths.py


1. Natural caption lengths for every panel captioner (ceiling 1024), plus
   Moondream default (normal) plus explicit `short` for comparison, on 16 pilot images.
2. SD35Medium with its T5 encoder loaded: generate from 4 natural-length
   captions, confirm no truncation warning, time per image.
Results -> analysis/natural_lengths.json.
"""

import base64, io, json, logging, sqlite3, sys, time, pathlib, re

sys.path.insert(0, "/home/ben/projects/panic_tda/priv/python")
import panic_models as pm
from PIL import Image

OUT = pathlib.Path(__file__).with_suffix(".json")
con = sqlite3.connect("/home/ben/projects/panic_tda/priv/panic_tda_dev.db")
rows = con.execute("""
  select i.output_image, i.id from invocations i join runs r on r.id=i.run_id
  join experiments e on e.id=r.experiment_id
  where e.i2t_max_new_tokens=1024 and e.num_runs=4 and i.type='image'
  and i.sequence_number in (0, 24, 48) order by i.id limit 16""").fetchall()
b64s = [base64.b64encode(r[0]).decode() for r in rows]
caps = [
    r[0]
    for r in con.execute("""
  select i.output_text from invocations i join runs r on r.id=i.run_id
  join experiments e on e.id=r.experiment_id
  where e.i2t_max_new_tokens=1024 and e.num_runs=4 and i.type='text'
  order by length(i.output_text) desc limit 4""")
]
print(
    f"{len(b64s)} images, {len(caps)} long captions ({[len(c.split()) for c in caps]} words)",
    flush=True,
)

pm.setup()
results = {}
ENDS = re.compile(r'[.!?"”)]$')


def summarise(texts, secs):
    words = sorted(len(t.split()) for t in texts)
    return {
        "n": len(texts),
        "median_words": words[len(words) // 2],
        "max_words": words[-1],
        "min_words": words[0],
        "pct_cut": 100 * sum(not ENDS.search(t.strip()) for t in texts) / len(texts),
        "secs_per_caption": secs / len(texts),
        "samples": texts[:2],
    }


for name in ["Moondream", "Gemma3n", "Qwen25VL", "Pixtral", "LLaMA32Vision"]:
    pm.load_model(name)
    pm.swap_to_gpu(name)
    t0 = time.time()
    texts = pm.invoke_i2t_batch(name, b64s)
    secs = time.time() - t0
    results[name] = summarise(texts, secs)
    print(
        name,
        json.dumps({k: v for k, v in results[name].items() if k != "samples"}),
        flush=True,
    )
    if name == "Moondream":
        model = pm._models["Moondream"]
        import torch

        texts = []
        t0 = time.time()
        with torch.inference_mode():
            for b in b64s:
                img = pm._decode_image_b64(b)
                texts.append(
                    model.caption(
                        model.encode_image(img),
                        length="short",
                        settings=pm._moondream_settings(),
                    )["caption"].strip()
                )
        results["Moondream(short)"] = summarise(texts, time.time() - t0)
        print(
            "Moondream(short)",
            json.dumps(
                {k: v for k, v in results["Moondream(short)"].items() if k != "samples"}
            ),
            flush=True,
        )
    pm.unload_model(name)
    OUT.write_text(json.dumps(results, indent=2))

# SD35Medium with T5
warnings_seen = []


class Catch(logging.Handler):
    def emit(self, record):
        warnings_seen.append(record.getMessage()[:200])


logging.getLogger("diffusers").addHandler(Catch())
pm.load_model("SD35Medium")
pm.swap_to_gpu("SD35Medium")
import torch

t0 = time.time()
_ = pm.invoke_t2i_batch("SD35Medium", caps)
secs = time.time() - t0
results["SD35Medium+T5"] = {
    "secs_per_image": secs / len(caps),
    "prompt_words": [len(c.split()) for c in caps],
    "truncation_warnings": [w for w in warnings_seen if "truncated" in w],
    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
}
print("SD35Medium+T5", json.dumps(results["SD35Medium+T5"]), flush=True)
OUT.write_text(json.dumps(results, indent=2))
print("DONE", flush=True)
