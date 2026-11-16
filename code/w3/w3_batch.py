"""Run several w3_sim.py commands in one guarded process; commands separated by '+':
  w3_batch.py cal 30 1000000 1 + cal 116 1000000 1"""
import sys, os, runpy, time
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
specs, cur = [], []
for tok in sys.argv[1:]:
    if tok == "+":
        specs.append(cur); cur = []
    else:
        cur.append(tok)
specs.append(cur)
for spec in specs:
    sys.argv = [os.path.join(HERE, "w3_sim.py")] + spec
    runpy.run_path(sys.argv[0], run_name="__main__")
    print(f"[batch] done: {' '.join(spec)} ({time.time()-t0:.1f}s)", flush=True)
