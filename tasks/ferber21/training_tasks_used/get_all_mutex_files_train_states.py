#!/usr/bin/env python3

import os
path_to_fd =  "../../../fast-downward.py" # Fill in with path to fast-downward.py on your install

for domainName in os.listdir():
    if os.path.isdir("{}/".format(domainName)):
        for problemName in os.listdir("{}/".format(domainName)):
            print("problem name is {}".format(problemName))
            if "domain" not in problemName and ".pddl" in problemName and ".mutexes" not in problemName and ".sas" not in problemName:
                os.system(
                    " {} --translate --sas-file {}/{}.sas {}/{}".format(path_to_fd, domainName, problemName, domainName, problemName))
