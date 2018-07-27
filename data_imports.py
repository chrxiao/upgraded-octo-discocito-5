import json

def load_json(path):
    with open(path) as json_data:
        d = json.load(json_data)
    return d
