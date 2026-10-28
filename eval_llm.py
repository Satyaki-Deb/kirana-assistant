"""Evaluates the message reader + catalog matcher on hand-labelled test messages.

Run without a key  -> scores the rule-based reader (baseline).
Run with a key     -> OPENAI_API_KEY=sk-... python eval_llm.py   (costs well under $0.05)

Each expected line = (expected SKU or "ASK" or "NOT_FOUND", qty, unit). A line is correct only if
the product decision, the quantity AND the unit are all right. "Hallucinated product" = the system
confidently picked a SKU when the right answer was ASK / NOT_FOUND, or picked a wrong SKU.
"""
import os, json, engine as E

TESTS = [
 ("2 peti parle g 50 gram", [("SKU0387", 2, "peti")]),
 ("tata salt 1 kg 1 peti", [("SKU0181", 1, "peti")]),
 ("5 peti lays magic masala 104 gm", [("SKU0449", 5, "peti")]),
 ("ek dabba fortune sunlite 15 litre", [("SKU0023", 1, "dabba")]),
 ("surf excel easy wash 1 kg 2 peti, red label 250 gm 1 peti", [("SKU0660", 2, "peti"), ("SKU0195", 1, "peti")]),
 ("bhaiya do peti kurkure masala munch 77 gm bhejo", [("SKU0443", 2, "peti")]),
 ("3 peti amul butter 100 gm aur 1 peti amul butter 500 gm", [("SKU0188", 3, "peti"), ("SKU0187", 1, "peti")]),
 ("aashirvaad atta 5 kilo 2 bori", [("ASK", 2, "bori")]),
 ("surf 2 peti", [("ASK", 2, "peti")]),
 ("1 dabba fortune", [("ASK", 1, "dabba")]),
 ("parle g 2 peti", [("ASK", 2, "peti")]),
 ("3 packet vim bar", [("NOT_FOUND", 3, "packet")]),
 ("2 peti horlicks chocolate biscuit xyz", [("NOT_FOUND", 2, "peti")]),
 ("10 packet colgate maxfresh 150 gm", [("ASK", 10, "packet")]),
 ("wagh bakri premium 250 gm 2 peti, haldiram aloo bhujia 150 gm 1 peti", [("SKU0190", 2, "peti"), ("SKU0431", 1, "peti")]),
 ("dettol liquid 550 ml ek peti", [("SKU0734", 1, "peti")]),
 ("saffola gold 1 litre 4 peti", [("SKU0030", 4, "peti")]),
 ("teen peti tata tea premium 250 gm", [("SKU0203", 3, "peti")]),
 ("parachute 200 ml 2 peti aur kissan ketchup 900 gm 1 peti", [("SKU0918", 2, "peti"), ("SKU0546", 1, "peti")]),
 ("good day cashew 200 gm 3 peti", [("SKU0374", 3, "peti")]),
]

def run(key=None):
    n = right = halluc = qty_ok = unit_ok = 0; fails = []; tokens = [0, 0]
    for msg, gold in TESTS:
        r = E.process(msg, None, key)
        if r.get("usage"): tokens[0] += r["usage"].get("prompt_tokens", 0); tokens[1] += r["usage"].get("completion_tokens", 0)
        got = [] if r.get("blocked") else r["results"]
        for i, (sku, q, u) in enumerate(gold):
            n += 1
            if i >= len(got): fails.append((msg, "line missing")); continue
            x = got[i]; d = x["status"].upper() if x["status"] != "ok" else x["sku_id"]
            qok, uok = x["line"]["qty"] == q, x["line"]["unit"] == u
            qty_ok += qok; unit_ok += uok
            if x["status"] == "ok" and d != sku: halluc += 1
            if d == sku and qok and uok: right += 1
            else: fails.append((msg, f"expected {sku} {q} {u} | got {d} {x['line']['qty']} {x['line']['unit']}"))
    print(f"engine={'LLM ' + os.environ.get('OPENAI_MODEL', 'gpt-4o-mini') if key else 'rule-based baseline'}")
    print(f"lines={n}  fully correct={right/n:.0%}  qty correct={qty_ok/n:.0%}  unit correct={unit_ok/n:.0%}  "
          f"wrong/hallucinated product picks={halluc}")
    if key: print(f"tokens in/out = {tokens[0]}/{tokens[1]}")
    for f in fails: print("  FAIL:", f)

if __name__ == "__main__":
    run(os.environ.get("OPENAI_API_KEY"))
