post='0119_tn_Pendulum-v1_eps1000000'
mode='tn'
envName='Pendulum-v1'
seed=0
batchSize=256
warmSteps=2500
maxSteps=1000000
evaluateFreq=5000

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python3 tn.py \
    --post ${post} \
    --mode ${mode} \
    --envName ${envName} \
    --seed ${seed} \
    --batchSize ${batchSize} \
    --warmSteps ${warmSteps} \
    --maxSteps ${maxSteps} \
    --evaluateFreq ${evaluateFreq} \
