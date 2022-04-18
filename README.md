# hopper-mpc-simple

## Table of Contents

- [Intro](#intro)
- [Setup](#setup)
- [Examples](#examples)

---
## Intro

This repository contains Python code for a simple simulation of a hopping point mass with model predictive control. The "simulator" uses RK4 integration (alternatively, you can directly use the DT system).

---

## Setup

1. Clone this directory wherever you want.

```shell 
git clone https://github.com/bbokser/hopper-mpc-simple.git
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
cd hopper-mpc-simple/src
python3.8 -m venv env
```
For more information on virtual environments: https://docs.python.org/3/library/venv.html
    
3. Activate the virtual environment, and then install numpy, scipy, matplotlib, sympy, cvxpy, and argparse.

```shell
source env/bin/activate
python3.8 -m pip install numpy scipy matplotlib cvxpy argparse
```
Don't use sudo here if you can help it, because it may modify your path and install the packages outside of the venv.

---

## Examples

Here is some example code:

```shell
cd hopper-mpc-simple/src
source env/bin/activate
python3.8 run.py 2 mpc
```
This simulates the "robot" in 2D with mpc. The output is a set of plots tracking the behavior over time.

To simulate with mpc in 3D:

```
python3.8 run.py 3 mpc
```

To simulate with trajectory optimized open loop control:

```
python3.8 run.py 2 openloop
```

```
python3.8 run.py 3 openloop
```



