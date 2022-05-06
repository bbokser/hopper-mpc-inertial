# hopper-mpc-inertial

## Table of Contents

- [Intro](#intro)
- [Setup](#setup)
- [Examples](#examples)

---
## Intro

This repository contains Python code for a simple simulation of a hopping rigid body with model predictive control. The simulator uses RK4 integration.

---

## Setup

1. Clone this directory wherever you want.

```shell 
git clone https://github.com/bbokser/hopper-mpc-inertial.git
```  

2. Make sure both Python 3.8 and pip are installed.

```shell
sudo apt install python3.8
sudo apt-get install python3-pip
python3.8 -m pip install --upgrade pip
```

2. I recommend setting up a virtual environment for this, as it requires the use of a number of specific Python packages.

```shell
sudo apt-get install python3.8-venv
cd hopper-mpc-inertial
python3.8 -m venv env
```
For more information on virtual environments: https://docs.python.org/3/library/venv.html
    
3. Activate the virtual environment, and then install numpy, scipy, matplotlib, sympy, cvxpy, and argparse.

```shell
source env/bin/activate
python3.8 -m pip install numpy scipy matplotlib cvxpy argparse tqdm casadi transforms3d
```
Don't use sudo here if you can help it, because it may modify your path and install the packages outside of the venv.

---

## Examples

Here is some example code:

```shell
cd hopper-mpc-inertial/src
source env/bin/activate
python3.8 run.py cvxpy
```
This simulates the "robot". The output is a set of plots tracking the behavior over time.



