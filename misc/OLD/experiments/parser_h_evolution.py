#! /usr/bin/env python

from lab.parser import Parser

import re

REF_H_EVOLUTION = "ref_h_evolution"
def parse_h_evolution(content, props):
    preamble = "Heuristic evolution during plan:"
    start = content.find(preamble)
    if start == -1:
        props[REF_H_EVOLUTION] = None
    else:
        data = {}
        round = 0
        start = content.find("\n", start) + 1
        # Next batch of lines
        while (content.startswith("Sum Costs", start) or content.startswith(";", start)) and start != -1:
            if content.startswith("Sum Costs", start):
                round += 1
            end = content.find("\n", start)
            line = content[start: end]
            line_split = line.split(";")
            for no, tech_value in enumerate(line_split):
                tech_value = tech_value.strip()
                if tech_value == "":
                    continue
                tech_value = tech_value.split(":")
                if len(tech_value) != 2:
                    props[REF_H_EVOLUTION] = None
                    props.add_unexplained_error(
                        "Other than 2 elements in Technique-Value pair")
                    return
                tech, value = tech_value[0].strip(), tech_value[1].strip()
                if tech == "" or value == "":
                    props[REF_H_EVOLUTION] = None
                    props.add_unexplained_error("Empty entry in technique-value pair")
                    return
                try:
                    value = int(value)
                except ValueError:
                    props.add_unexplained_error("Invalid heuristic value in technique-value pair")
                    props[REF_H_EVOLUTION] = None
                    return

                if tech not in data:
                    if round > 1:
                        props.add_unexplained_error("Changing number of techniques in h evolution output.")
                        props[REF_H_EVOLUTION] = None
                        return
                    data[tech] = []
                data[tech].append(value)
            start = end + 1


        props[REF_H_EVOLUTION] = data


print 'Running h evolution parser'
parser = Parser()
parser.add_function(parse_h_evolution)
parser.parse()
