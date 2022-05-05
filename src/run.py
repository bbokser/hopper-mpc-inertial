"""
Copyright (C) 2021 Benjamin Bokser
"""
from robotrunner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("tool", help="'choose cvxpy or casadi",
                    choices=['cvxpy', 'casadi'],
                    type=str)

parser.add_argument("--runtime", help="sim run time in ms (integer)", type=int, default=5000)

args = parser.parse_args()

dt = 1e-3

runner = Runner(dt=dt, tool=args.tool, t_run=args.runtime)
runner.run()
