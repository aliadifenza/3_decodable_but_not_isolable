import json, runpy, sys, copy
sys.argv=["x"]; g = runpy.run_path(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "test_g5r_dataset_local.py"), run_name="__main__")
GR, R = g["GR"], g["R"]
def wrap(R): return dict(data_sha256="h", identity=dict(model="m", revision="r", sae_files={}), runtime_s=1, gpu_cost_usd_estimate=0.1, results=R)
A = copy.deepcopy(R); B = copy.deepcopy(R)
A["layers"] = {"2": R["layers"][2]}; A["layer_verdicts"] = {"2": R["layer_verdicts"][2]}
B["layers"] = {"3": R["layers"][3]}; B["layer_verdicts"] = {"3": R["layer_verdicts"][3]}
json.dump(wrap(A), open("/tmp/a.json","w"), default=str); json.dump(wrap(B), open("/tmp/b.json","w"), default=str)
import merge_g5r; merge_g5r.main("/tmp/m.json", "/tmp/a.json", "/tmp/b.json")
M = json.load(open("/tmp/m.json")); assert M["layers_covered"] == [2, 3] and not M["complete"] and M["L_star"] is None; print("MERGE TEST PASS")
