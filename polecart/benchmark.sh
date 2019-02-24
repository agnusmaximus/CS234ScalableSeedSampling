#!/bin/bash

python3.6 policy_gradient_cartpole_single.py 1 > 1_out; python3.6 policy_gradient_cartpole_single.py 2 > 2_out; python3.6 policy_gradient_cartpole_single.py 4 > 4_out; python3.6 policy_gradient_cartpole_single.py 8 > 8_out;
