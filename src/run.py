"""
Copyright (C) 2021 Benjamin Bokser
"""
from robotrunner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dyn", help="'choose euler or se3",
                    choices=['euler', 'se3'],
                    type=str)
args = parser.parse_args()

dt = 1e-3

runner = Runner(dt=dt, dyn=args.dyn)
runner.run()
