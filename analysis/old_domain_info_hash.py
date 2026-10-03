"""
Prints SHA-256 hashes of the OLD (non-ternary) domain's step info dicts and Monitor CSV
for one graph convention, so the old domain's info and Monitor output can be checked as
byte-identical across commits (tests/test_ternary_diagnostic.py's TestOldDomainInfoUnchanged).

Same env, seed and RNG-free action rule as analysis/old_domain_obs_hash.py (imported from
it as a sibling file, so copy both files together), wrapped in the SB3 fork's Monitor with
gnn_global.py's old-domain info_keywords ("len100", "len200"). The env's timeout is
shortened to TIMEOUT so several episodes end inside NUM_STEPS and the Monitor CSV gets
real rows. Time-dependent fields (the CSV's t_start and "t" column, info["episode"]["t"])
are dropped before hashing; everything else is hashed as written.

Imports only things that exist at commit a628f16. Run from the repo root:

    PYTHONPATH=. python analysis/old_domain_info_hash.py oracle_sage

Prints "monitor <sha256>" and "info <sha256>" on stdout; the stack goes to stderr.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from old_domain_obs_hash import CONVENTIONS, _install_randint_shim, sample_action, stack_key  # noqa: E402

import gym  # noqa: E402

SEED = 0
NUM_STEPS = 300
TIMEOUT = 100
OLD_DOMAIN_INFO_KEYWORDS = ("len100", "len200")


def _normalised_monitor_csv(path):
    """The Monitor file minus its time-dependent parts: t_start in the leading JSON
    comment line, and the "t" column."""
    with open(path) as f:
        comment = json.loads(f.readline()[1:])
        table = list(csv.reader(f))
    comment.pop("t_start", None)
    keep = [i for i, name in enumerate(table[0]) if name != "t"]
    out = io.StringIO()
    out.write("#" + json.dumps(comment, sort_keys=True) + "\n")
    for row in table:
        out.write(",".join(row[i] for i in keep) + "\n")
    return out.getvalue()


def _normalised_info(info):
    info = dict(info)
    if "episode" in info:
        episode = dict(info["episode"])
        episode.pop("t", None)
        info["episode"] = episode
    return json.dumps(info, sort_keys=True, default=repr)


def old_domain_info_hashes(convention, info_keywords=OLD_DOMAIN_INFO_KEYWORDS,
                           seed=SEED, num_steps=NUM_STEPS, timeout=TIMEOUT):
    """:return: (monitor_sha256, info_sha256)"""
    from sage.domains.gym_taxi import REWARDS
    from sage.domains.gym_taxi.envs.taxi_env import GraphTaxiEnv
    from sage.forks.stable_baselines3.stable_baselines3.common.monitor import Monitor

    env = GraphTaxiEnv(
        representation="graph", scenario="city", mask=False,
        rewards=REWARDS["v1"], graph_convention=convention,
    )
    env.scenario = dict(env.scenario, timeout=timeout)  # a copy: the shared SCENARIOS dict is untouched
    env.seed(seed)
    monitor_dir = tempfile.mkdtemp()
    try:
        menv = Monitor(env, filename=monitor_dir, info_keywords=tuple(info_keywords))
        menv.reset()
        infos = []
        for _ in range(num_steps):
            _obs, _reward, done, info = menv.step(sample_action(env.sim))
            infos.append(_normalised_info(info))
            if done:
                menv.reset()
        menv.close()
        monitor_text = _normalised_monitor_csv(os.path.join(monitor_dir, Monitor.EXT))
    finally:
        shutil.rmtree(monitor_dir, ignore_errors=True)
    monitor_hash = hashlib.sha256(monitor_text.encode()).hexdigest()
    info_hash = hashlib.sha256("\x00".join(infos).encode()).hexdigest()
    return monitor_hash, info_hash


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("convention", choices=CONVENTIONS)
    args = parser.parse_args(argv)

    _install_randint_shim()
    sys.stderr.write("stack={} (gym {}) convention={} seed={} steps={} timeout={}\n".format(
        stack_key(), gym.__version__, args.convention, SEED, NUM_STEPS, TIMEOUT))
    monitor_hash, info_hash = old_domain_info_hashes(args.convention)
    print("monitor " + monitor_hash)
    print("info " + info_hash)


if __name__ == "__main__":
    main(sys.argv[1:])
