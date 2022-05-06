"""
Copyright (C) 2021 Benjamin Bokser
"""
from robotrunner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dyn", help="choose 2f or 3f",
                    choices=['2f', '3f'],
                    type=str)

parser.add_argument("--curve", help="make the ref traj curved", action="store_true")

parser.add_argument("--runtime", help="sim run time in ms (integer)", type=int, default=5000)

args = parser.parse_args()

if args.curve:
    curve = True
else:
    curve = False

dt = 1e-3

runner = Runner(dt=dt, dyn=args.dyn, curve=curve, t_run=args.runtime)
runner.run()
