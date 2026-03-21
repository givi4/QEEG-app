import json
from pathlib import Path

channels = [
    "FP1","FP2","F7","F3","FZ","F4","F8",
    "T7","C3","CZ","C4","T8",
    "P7","P3","PZ","P4","P8",
    "O1","O2"
]

norms = {
    "metadata": {
        "version": "1.1",
        "source": "placeholder — populate with real normative values",
        "power_type": "relative",
        "bands": ["Delta","Theta","Alpha","Beta","HiBeta","Gamma"]
    },
    "norms": {
        ch: {
            "Delta":  {"mean": 0.28, "std": 0.08},
            "Theta":  {"mean": 0.18, "std": 0.06},
            "Alpha":  {"mean": 0.30, "std": 0.09},
            "Beta":   {"mean": 0.10, "std": 0.04},
            "HiBeta": {"mean": 0.04, "std": 0.02},
            "Gamma":  {"mean": 0.03, "std": 0.02},
        }
        for ch in channels
    }
}

Path("normative_data").mkdir(exist_ok=True)
with open("normative_data/placeholder_norms.json", "w") as f:
    json.dump(norms, f, indent=2)

print("Done — normative_data/placeholder_norms.json rebuilt with 6 bands.")