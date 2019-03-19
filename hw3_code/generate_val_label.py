#!/bin/python

import numpy
import os
import sys

if __name__ == '__main__':
    events = ["NULL"]  # one of P001, P002, P00

    for event in events:
        fread = open("list/val", "r")
        fwrite = open("list/" + event + "_val_label", "wb")

        for line in fread.readlines():
            file_name, eve = line.replace('\n', '').split()
            if eve == event:
                fwrite.write("1\n")
            else:
                fwrite.write("0\n")
