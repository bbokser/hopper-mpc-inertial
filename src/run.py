"""
Copyright (C) 2021 Benjamin Bokser
"""
from robotrunner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("ctrl", help="'choose mpc or openloop",
                    choices=['mpc', 'openloop'],
                    type=str)
args = parser.parse_args()

dt = 1e-3

runner = Runner(dt=dt, ctrl=args.ctrl)
runner.run()
