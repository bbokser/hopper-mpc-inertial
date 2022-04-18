"""
Copyright (C) 2021 Benjamin Bokser
"""
from robotrunner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dims", help="choose the dimensionality: 2d or 3d",
                    choices=[2, 3], type=int)
parser.add_argument("ctrl", help="'choose mpc or openloop",
                    choices=['mpc', 'openloop'],
                    type=str)
args = parser.parse_args()

dt = 1e-3

runner = Runner(dt=dt, dims=args.dims, ctrl=args.ctrl)
runner.run()
