#!/bin/bash

for i in `seq 1 10`;
do
    # Iter 1
    python3.6 policy_gradient_cartpole_single.py 1 > 1_out_$i
    python3.6 policy_gradient_cartpole_single.py 2 > 2_out_$i
    python3.6 policy_gradient_cartpole_single.py 4 > 4_out_$i
    python3.6 policy_gradient_cartpole_single.py 8 > 8_out_$i
done
