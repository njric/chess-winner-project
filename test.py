from chess.utils import *
import hashlib
import json
import pandas as pd

list_pickles = list_pickles('pickle')

baseline = {}
count = 0
for pickle in list_pickles:
    data = from_disk(pickle)
    old, action, rwd, new = data[0]
    #print(old)
    dhash = hashlib.sha256()
    old_list = old['observation'].tolist()
    for i in old_list:
        encoded = json.dumps(i, sort_keys=True).encode()
        hashed = hash(encoded)
        if hashed in baseline.keys():
            temp = baseline[hashed]
            if action in temp.keys():
                baseline[hashed][action] = baseline[hashed][action]+1
            else:
                baseline[hashed][action] = 1
        else:
            baseline[hashed] = {action : 1}
    count += 1
    if count == 50:
        print(baseline)
        exit()
