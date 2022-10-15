from sys import argv

for file in argv[1:]:
    with open(file) as f:
        data = f.read()
    data = data\
        .replace("transport", "transportunit")\
        .replace("scanalyzer", "scanalyzerunit")\
        .replace("/transportunit/transportunit/", "/transport/transport/")\
        .replace("/scanalyzerunit/scanalyzerunit/", "/scanalyzer/scanalyzer/")
    with open(file.replace("transport", "transportunit").replace("scanalyzer", "scanalyzerunit"), "w") as f:
        f.write(data)
