#!/usr/bin/env python3

from sys import argv

epoches = int(argv[1])
print("name", "network seed", sep=",", end="")
for i in range(epoches):
    print(f",train_loss_e{i}", end="")
for i in range(epoches):
    print(f",val_loss_e{i}", end="")
print()
for result in argv[2:]:
    with open(result+"/nfd.log",) as f:
        log = f.readlines()
    train_loss, val_loss = [], []
    for line in log:
        if "avg_train_loss" in line:
            epoch = int(line.split("Epoch ")[1].split(" ")[0])
            if epoch >= epoches:
                break
            train_loss.append(line.split("avg_train_loss=")[1].split(" ")[0])
            if "avg_val_loss" in line:
                val_loss.append(line.split("avg_val_loss=")[1].split("\n")[0])
    if result[-1] == "/":
        result = result[:-1]
    name, seed = result.split("/")[-1].split(".")[1:3]
    seed = seed.replace("ns", "")

    print(name, seed, sep=",", end="")
    if len(train_loss + val_loss) == 2*epoches:
        for loss in train_loss:
            print(","+loss, end="")
        for loss in val_loss:
            print(","+loss, end="")
        print()
    else:
        print(",,,,,,,,,,")
