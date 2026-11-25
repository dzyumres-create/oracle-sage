"""
Run a script with gym 0.18's seeding installed as gym.utils.seeding.np_random (on gym >= 0.26).

The envs on this branch seed through sage/domains/gym_taxi/utils/compat.np_random, which on
gym >= 0.26 is gym 0.18's algorithm, so a seed gives RCP's episodes. Older commits (e.g. the
ebbf25b reference for the old-domain hash tests) call seeding.np_random directly; this wrapper
makes them seed the same way, so their reference hashes can be recomputed on the Mac.
compat.py is loaded by file path from THIS checkout, so it works against any other checkout.

Run from the root of the checkout under test:
    PYTHONPATH=. python /path/to/paper-unified/analysis/with_gym018_seeding.py \
        /path/to/old_domain_obs_hash.py atom
"""
import importlib.util
import os
import runpy
import sys

import gym.utils.seeding as seeding

HERE = os.path.dirname(os.path.abspath(__file__))
COMPAT = os.path.join(HERE, "..", "sage", "domains", "gym_taxi", "utils", "compat.py")


def main(argv):
    spec = importlib.util.spec_from_file_location("_compat_gym018", COMPAT)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    if compat.LEGACY_GYM:
        sys.stderr.write("gym < 0.26: its own seeding already is gym 0.18's; nothing installed\n")
    else:
        seeding.np_random = compat.np_random  # gym 0.18's algorithm; never calls seeding itself
    sys.argv = list(argv)
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
