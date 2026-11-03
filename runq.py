"""Run a list of settings on low priority without fighting other jobs. Finished settings are skipped, so a stopped list continues by running the same command.

Each worker uses one thread and builds a world once. A worker waits while the machine's one-minute load is above LOAD. Speeds from these runs are not claims; speed.py measures one run at a time.
Workers are spawned, not forked, because torch on macOS crashes in a forked child. A worker that dies raises here instead of hanging the list.
"""
import concurrent.futures as cf, multiprocessing as mp, os, time

LOAD = float(os.environ.get("LOAD", 14)); PAR = int(os.environ.get("PAR", 3)); WORLDS = {}; EX = None
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"


def job(a):
    import arms, env
    if os.getpriority(os.PRIO_PROCESS, 0) < 20: os.nice(20)
    while os.getloadavg()[0] > LOAD: time.sleep(3)
    k = (a.tier, a.world, a.steps, a.double); c = dict(env.TIERS[a.tier]); steps = (a.steps or c["steps"]) * (2 if a.double else 1)
    if k not in WORLDS: WORLDS.clear(); WORLDS[k] = env.World(c, steps, a.world)
    try: return env.train(a, arms.parse(a.spec), WORLDS[k], os.path.join(env.HERE, a.results))
    except Exception as e: return dict(error=repr(e), spec=a.spec)


def run(jobs, seconds=0, par=0):
    """jobs are namespaces from env.parser(). Returns how many are left when the time runs out, so the caller can say so instead of waiting."""
    import arms, env
    path = os.path.join(env.HERE, jobs[0].results) if jobs else ""; done = env.done_keys(path); seen = set(); todo = []
    for a in jobs:
        k = env.key_of(a, arms.parse(a.spec))
        if k not in done and k not in seen: seen.add(k); todo.append(a)
    if not todo: return 0
    global EX
    EX = EX or cf.ProcessPoolExecutor(par or PAR, mp_context=mp.get_context("spawn")); fs = [EX.submit(job, a) for a in todo]
    try:
        for f in cf.as_completed(fs, timeout=seconds or None):
            r = f.result()
            if r.get("error"): print("ERROR", r["spec"][:80], r["error"], flush=True)
    except cf.TimeoutError:
        EX.shutdown(wait=True, cancel_futures=True); EX = None
    done = env.done_keys(path); return sum(env.key_of(a, arms.parse(a.spec)) not in done for a in todo)
