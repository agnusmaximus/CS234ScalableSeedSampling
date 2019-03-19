#!/bin/bash

# Reweight
for i in `seq 1 4`;
do
#    python3.6 policy_gradient_cartpole_single.py 1 1 > 1_out_reweight_$i
#    python3.6 policy_gradient_cartpole_single.py 2 1 > 2_out_reweight_$i
#    python3.6 policy_gradient_cartpole_single.py 4 1 > 4_out_reweight_$i
    python3.6 policy_gradient_cartpole_single.py 8 1 > 8_out_reweight_$i
done

# No Reweight
for i in `seq 1 4`;
do
#    python3.6 policy_gradient_cartpole_single.py 1 0 > 1_out_$i
    #python3.6 policy_gradient_cartpole_single.py 2 0 > 2_out_$i
    #python3.6 policy_gradient_cartpole_single.py 4 0 > 4_out_$i
    python3.6 policy_gradient_cartpole_single.py 8 0 > 8_out_$i
done
