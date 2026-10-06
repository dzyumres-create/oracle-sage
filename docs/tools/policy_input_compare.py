"""Compare two policy_input_harness.py outputs record-by-record. usage: policy_input_compare.py REF.json NEW.json"""
import json, sys

ref, new = (json.load(open(p)) for p in sys.argv[1:3])
mism = []
assert ref["goals"] == new["goals"], "goal sequences differ"
assert len(ref["records"]) == len(new["records"])
n_tensors = 0
fields_seen = set()
for dec, (r, n) in enumerate(zip(ref["records"], new["records"])):
    for key in ("live_json", "plans"):
        if r[key] != n[key]:
            mism.append((dec, key))
    for i, (a, b) in enumerate(zip(r["live"], n["live"])):
        for f in a:
            fields_seen.add(f)
            n_tensors += a[f] is not None
            if a[f] != b[f]:
                mism.append((dec, "live", i, f))
    for i, (pa, pb) in enumerate(zip(r["proj"], n["proj"])):
        for j, (a, b) in enumerate(zip(pa, pb)):
            for f in a:
                n_tensors += a[f] is not None
                if a[f] != b[f]:
                    mism.append((dec, "proj", i, j, f))
if ref["ends"] != new["ends"]:
    mism.append(("ends",))
present = sorted(f for f in fields_seen if any(rec["live"][0][f] is not None for rec in ref["records"][:1]))
print(f"decisions={len(ref['records'])} envs={len(ref['records'][0]['live'])} tensors_compared={n_tensors} "
      f"fields_present={present}")
print(f"stats ref={ref['stats']}")
print(f"episode ends={len(ref['ends'])} mid_plan={sum(e['mid_plan'] for e in ref['ends'])} "
      "[" + ", ".join("env%d@dec%d step %d/%d" % (e["env"], e["decision"], e["plan_step"] + 1, e["plan_length"]) for e in ref["ends"]) + "]")
print(f"json built: ref={ref['json_built']} new={new['json_built']}  obs_mask used: ref={ref['supports_obs_mask']} new={new['supports_obs_mask']}")
print(f"wall: ref={ref['wall']:.1f}s new={new['wall']:.1f}s")
print("RESULT:", "BYTE-IDENTICAL" if not mism else f"{len(mism)} MISMATCHES, first: {mism[:5]}")
