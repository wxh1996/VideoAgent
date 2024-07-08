import json

data = json.load(open("egochema_subset_5cap_selfevalCoT_step3_recap_eva448_newcap_v14_allfeat_subset_final.json"))
print(len(data))

accs = []
frames = []
for key in data:
    acc = data[key][1]
    accs.append(acc)
    
    frame = data[key][2]
    frames.append(frame)

print("Mean accuracy: ", sum(accs)/len(accs))
print("Mean frame: ", sum(frames)/len(frames))
