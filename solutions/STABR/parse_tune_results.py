import glob
import json
import pprint

trials = [p for p in glob.glob("*/trial.json")]

results = {}
for trial in trials:
    with open(trial) as source:
        data = json.load(source)
    if data["score"]:
        results[data["score"]] = data["hyperparameters"]["values"]
        
topfive = sorted(results.keys(), reverse=True)[:5]

for entry in topfive:
    print(entry)
    pprint.pprint(results[entry])
    print("\n")