from sys import argv

for file in argv[1:]:
    with open(file) as f:
        data = f.read()
    for p in ["2.00", "2.50", "3.00"]:
        with open(file.replace("-1-25.json", "-"+p.replace(".", "-")+".json"), "w") as f:
            f.write(
                data.replace('-1-25",', '-'+p.replace('.', '-')+'",').replace('"bound-multiplier": 1.25', f'"bound-multiplier": {float(p)}')
            )
