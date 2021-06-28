#!/bin/bash

# VARIABLES
TIME=4
CORES=8
GPU_MODEL="GeForceRTX2080Ti"
MEMORY="7892"
DATE=$(date +'D%_d-%m-%y-T%H-%M-%S')
NAME="temp"
# Hidden variables

_help_description="Script to launch experiments for the NIPS proposal.
bash launcher.sh <experiment_code> --options
<experiment_code>
    A1 :nips_A1_experiment.py
OPTIONS:
    --time      : time in hours for the queue. Default is 4 hrs
    --cores     : cpu cores to reserve. Default is 8.
    --gpu_model : GPU model. 
                [GeForceRTX2080Ti, GeForceGTX1080Ti, GeForceGTX1080, TeslaV100_SXM2_32GB]
    --memory    : Memeory to reserve. Defalut is 7892.
"

# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : augmentation
# $4 : GPU model
launch_experimentA1() {
    # Launching on Cluster.
    echo "Augmentation: $3 "
    # echo " bsub -J A1_$3 -W $1:00 \-o /cluster/scratch//adahiya/nipsa1_$3_logs.out \
    # -n $2 -R 'rusage[mem=7892, ngpus_excl_p=1 gpu_model0==$4]' \
    # -G ls_infk \
    # python src/experiments/NIPS/nips_A1_experiment.py $3"
    bsub -J "A1_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/nipsa1_$3_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$4]" \
        -G ls_infk \
        python src/experiments/NIPS/nips_A1_experiment.py $3

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : augmentation
# $4 : GPU model
launch_experimentA2() {
    # Launching on Cluster.
    echo "Augmentation: $3 "
    bsub -J "A2_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/nipsa2_$3_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$4]" \
        -G ls_infk \
        python src/experiments/NIPS/nips_A2_experiment.py $3

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
# $4 : experiment_key
# $5 : experiment_name
# $6 : experiment_type
launch_experimentA_downstream() {
    # Launching on Cluster.
    echo "Experiment $5 downstream experiment submitted!"
    bsub -J "A1_$5" -W "$1:00" \-o "/cluster/scratch//adahiya/ssl_$5_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$3]" \
        -G ls_infk \
        python src/experiments/NIPS/downstream_experiment.py $4 $5 $6

}

# To launch Imagenet encoder for downstream.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
launch_imagenet_downstream() {
    # Launching on Cluster.
    echo "Experiment Imagenet downstream experiment submitted!"
    bsub -J "imagenet_downstream" -W "$1:00" \-o "/cluster/scratch//adahiya/ssl_imagenet_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$3]" \
        -G ls_infk \
        python src/experiments/NIPS/downstream_experiment.py imagenet imagenet IMAGENET

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
# $4 : epochs
# $5 : accumulate_grad_batches
launch_experimentB() {
    # Launching on Cluster.
    command=" -n $2 -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
            -R  'select[gpu_model0==$3]' -G ls_infk \
            python src/experiments/hybrid1_experiment.py -batch_size 512 \
            -accumulate_grad_batches $5 -epochs $4 -tag NIPS_B"
    declare -a contrastive_augment=("-contrastive color_jitter"
        "")
    declare -a pairwise_augment=("-pairwise rotate"
        "-pairwise crop"
        "-pairwise crop -pairwise rotate")
    K=1
    for i in "${contrastive_augment[@]}"; do
        for j in "${pairwise_augment[@]}"; do
            # echo "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
            #         $command $j $i"
            eval "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
                    $command $j $i"
            K=$(($K + 1))
        done
    done
    # echo "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
    #                 $command "-contrastive color_jitter" $i"
    eval "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
                    $command "-contrastive color_jitter" $i"
}
if [ $# -eq 0 ]; then
    echo "No Experiment selected!"
elif [[ $1 == "-h" ]]; then
    echo "$_help_description"
    exit 0
else
    EXPERIMENT="$1"
    echo $EXPERIMENT
    shift
    while [ -n "$1" ]; do
        case "$1" in
        --time)
            TIME=$2
            shift
            ;;
        --cores)
            CORES=$2
            shift
            ;;
        --gpu_model)
            GPU_MODEL=$2
            shift
            ;;
        --memory)
            MEMORY=$2
            shift
            ;;
        --name)
            NAME=$2
            shift
            ;;
        *)
            echo "Option $1 not recognized"
            echo "(Run $0 -h for help)"
            exit -1
            ;;
        esac
        shift
    done
fi

# $1: additional args
launch_simclr() {
    bsub -J "simclr" -W "$TIME:00" \-o "/cluster/scratch//adahiya/simclr_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/simclr_experiment.py $1
    # python src/experiments/simclr_experiment.py $1
}

# $1: additional args
launch_pairwise() {
    bsub -J "pair" -W "$TIME:00" \-o "/cluster/scratch//adahiya/pair_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/pairwise_experiment.py $1
    # python src/experiments/pairwise_experiment.py $1
}

# $1: additional args
launch_semisupervised() {
    bsub -J "ssl" -W "$TIME:00" \-o "/cluster/scratch//adahiya/ssl_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/semi_supervised_experiment.py $1
    # python src/experiments/semi_supervised_experiment.py $1
}

# $1: additional args
launch_hybrid1() {
    bsub -J "hybrid1" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid1_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid1_experiment.py $1
    # python src/experiments/hybrid1_experiment.py $1

}

# $1: additional args
launch_hybrid2() {
    bsub -J "hybrid2" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid2_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid2_experiment.py $1
    # python src/experiments/hybrid2_experiment.py $1

}

# $1 : args
launch_supervised() {
    bsub -J "supervised" -W "$TIME:00" \-o "/cluster/scratch//adahiya/supervised_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/baseline_experiment.py $1

}

seed1=5
seed2=15
seed3=25
# Main process
case $EXPERIMENT in
SIM_ABL)
    meta_file="simclr_ablative"
    echo "Launching Simclr ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed3" "$SAVED_META_INFO_PATH/${meta_file}$seed3.bkp.$DATE"
    declare -a augmentations=("color_drop"
        "color_jitter"
        "crop"
        "cut_out"
        "gaussian_blur"
        "random_crop"
        "rotate"
        "gaussian_noise"
        "sobel_filter")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 -train_ratio 0.9\
         -sources freihand  -tag sim_abl -tag pmlr  -save_top_k 1  -save_period 1 -resnet_size 18 -num_worker $CORES"
    # For larger Resnets
    # args="--resize  -batch_size 128 -epochs 100 -accumulate_grad_batches 16 -train_ratio 0.9\
    #      -sources freihand  -tag sim_abl -tag iccv  -save_top_k 1  -save_period 1 -resnet_size 50  -num_worker $CORES"
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_simclr " --$j  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    done
    # for j in "${augmentations[@]}"; do
    #     echo "$j $seed2"
    #     launch_simclr " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    # done
    ;;
SIM_ABL_DOWN)
    echo "Launching downstream simclr ablative studies"
    meta_file="simclr_ablative_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed3" "$SAVED_META_INFO_PATH/${meta_file}$seed3.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam -train_ratio 0.9 \
         -sources freihand -tag sim_abl -tag pmlr -resnet_size 18 -num_worker $CORES  -save_top_k 1  -save_period 1"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/simclr_ablative$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/simclr_ablative$seed2
    ;;
PAIR_ABL_FH)
    meta_file="pair_ablative_fh"
    echo "Launching Pair ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("color_jitter"
        "crop"
        "rotate"
        "random_crop")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 -train_ratio 0.9\
         -sources freihand  -tag pair_abl -tag pmlr -tag pair_comp -save_top_k 1  -save_period 1 -resnet_size 18 -num_worker $CORES"
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed1 -seed $seed1"
    done
    # for j in "${augmentations[@]}"; do
    #     echo "$j $seed2"
    #     launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    # done
    ;;
PAIR_ABL_FH_DOWN)
    echo "Launching Pairwise ablative studies"
    meta_file="pair_ablative_fh_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag pair_abl -tag pmlr -tag pair_comp -resnet_size 18 -save_top_k 1  -save_period 1 "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/pair_ablative_fh$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/pair_ablative_fh$seed2
    ;;
PAIR_COMPOSITION_SEARCH)
    meta_file="pair_comp"
    echo "Launching Pair exhaustive studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("--color_jitter --rotate --crop --random_crop"
    "--color_jitter --rotate --crop"
    "--color_jitter --rotate --random_crop"
    "--color_jitter --crop --random_crop"
    "--rotate --crop --random_crop"
    "--color_jitter --rotate"
    "--color_jitter --crop"
    "--color_jitter --random_crop"
    "--rotate --random_crop"
    "--rotate --crop"
    "--crop --random_crop"
    )
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 -train_ratio 0.9 -resnet_size 18\
         -sources freihand  -tag pair_comp -tag pmlr  -save_top_k 1  -save_period 1 -resnet_size 18 -num_worker $CORES"
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_pairwise " $j  $args  -meta_file ${meta_file}$seed1 -seed $seed1"
    done
    # for j in "${augmentations[@]}"; do
    #     echo "$j $seed2"
    #     launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    # done
    ;;
PAIR_COMPOSITION_SEARCH_DOWN)
    echo "Launching Pairwise ablative studies"
    meta_file="pair_comp_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag pair_comp -tag pmlr -resnet_size 18 -save_top_k 1  -save_period 1 "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/pair_comp$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/pair_ablative_ih$seed2
    ;;
SIM_COMPOSITION_SEARCH)
    meta_file="sim_comp"
    echo "Launching simclr exhaustive studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("--color_jitter --rotate --crop --random_crop"
    "--color_jitter --rotate --crop"
    "--color_jitter --rotate --random_crop"
    "--color_jitter --crop --random_crop"
    "--rotate --crop --random_crop"
    "--color_jitter --rotate"
    "--color_jitter --crop"
    "--color_jitter --random_crop"
    "--rotate --random_crop"
    "--rotate --crop"
    "--crop --random_crop"
    )
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 -train_ratio 0.9 -resnet_size 18\
         -sources freihand  -tag sim_comp -tag pmlr  -save_top_k 1  -save_period 1 -resnet_size 18 -num_worker $CORES"
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_simclr " $j  $args  -meta_file ${meta_file}$seed1 -seed $seed1"
    done
    # for j in "${augmentations[@]}"; do
    #     echo "$j $seed2"
    #     launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    # done
    ;;
SIM_COMPOSITION_SEARCH_DOWN)
    echo "Launching Simclr exhaustive comp studies"
    meta_file="sim_comp_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag sim_comp -tag pmlr -resnet_size 18"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/sim_comp$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/sim_comp$seed2
    ;;
CROSS_DATA_SIM)
    # NOTE: Remember to set the warmup epoch as 3 and checkpoint saving on train set.
    echo "Launching sim cross dataset"
    meta_file="sim_crossdataset"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="-sources freihand -sources interhand --color_jitter --random_crop --crop -warmup_epochs 3 -epochs 26 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag sim_cross -tag pmlr -train_ratio 0.999999999999"
    launch_simclr "$args -meta_file $meta_file$seed1 -seed $seed1"
    # launch_simclr "$args -meta_file $meta_file$seed2 -seed $seed2"
    ;;
CROSS_DATA_SIM_DOWN)
    echo "Launching Simclr exhaustive comp studies"
    meta_file="sim_crossdataset_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -optimizer adam \
         -sources freihand -save_top_k 1  -save_period 1  -tag sim_cross -tag pmlr -resnet_size 18 -train_ratio 0.999999999999999"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 --encoder_trainable"
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 -tag frozen"
    done <$SAVED_META_INFO_PATH/sim_crossdataset$seed1
    ;;
SIM_HEATMAP)
    # NOTE: Remember to set the warmup epoch as 3 and checkpoint saving on train set.
    echo "Launching simclr heatmap based model."
    meta_file="sim_heatmap"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--heatmap --color_jitter --random_crop --crop  -batch_size 128 \
     -accumulate_grad_batches 16 -save_top_k 1  -save_period 1 -tag sim_heatmap -tag pmlr -train_ratio 0.999999999999"
    launch_simclr "$args -meta_file $meta_file$seed1 -seed $seed1 -sources freihand -sources interhand -warmup_epochs 3 -epochs 26 -tag ih_fh"
    launch_simclr "$args -meta_file $meta_file$seed1 -seed $seed1 -sources freihand -warmup_epochs 10 -epochs 100 -tag fh"
    launch_simclr "$args -meta_file $meta_file$seed1 -seed $seed1 -sources interhand -warmup_epochs 4 -epochs 35 -tag ih"
    # launch_simclr "$args -meta_file $meta_file$seed2 -seed $seed2"
    ;;
SIM_HEATMAP_DOWN)
    echo "Launching Simclr exhaustive comp studies"
    meta_file="sim_heatmap_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args=" --heatmap --rotate --crop --resize  -batch_size 128 -optimizer adam \
     -tag sim_cross -tag pmlr -train_ratio 0.999999999999999 -save_top_k 1  -save_period 1"
    launch_semisupervised "$args -experiment_key <ADD_key> -experiment_name simclr_heat_ih_fh -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 -sources freihand --encoder_trainable"
    launch_semisupervised "$args -experiment_key <ADD_key> -experiment_name simclr_heat_ih_fh -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 --encoder_trainable"
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 --encoder_trainable"
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -epochs 100 -tag frozen"
    # done <$SAVED_META_INFO_PATH/sim_crossdataset$seed1

HYB1_ABL)
    echo "Launching hybrid1 ablative studies"
    meta_file="hybrid1_ablative"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand -resnet_size 18 -tag hyb1_abl -save_top_k 1  -save_period 1 "
    declare -a seeds=($seed1
        $seed2
    )
    declare -a contrastive_augment=("-contrastive color_jitter "
        " ")
    declare -a pairwise_augment=("-pairwise rotate "
        "-pairwise crop "
        "-pairwise crop -pairwise rotate ")
    for seed in "${seeds[@]}"; do
        for i in "${contrastive_augment[@]}"; do
            for j in "${pairwise_augment[@]}"; do
                launch_hybrid1 " $args -meta_file $meta_file$seed  $i $j -seed $seed"
            done
        done
    done
    ;;
CROSS_DATA_HYB1)
    echo "Launching hybrid 1 cross dataset"
    meta_file="hybrid1_crossdataset"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="-sources freihand -sources interhand -contrastive color_jitter -pairwise crop -pairwise rotate -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag hyb1_cross"
    launch_hybrid1 "$args -meta_file $meta_file$seed1 -seed $seed1"
    launch_hybrid1 "$args -meta_file $meta_file$seed2 -seed $seed2"
    ;;
SEMISUPERVISED)
    args=" --rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -num_workers $CORES  "
    echo "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
        -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
        -R 'select[gpu_model0==$GPU_MODEL]'  \
        -G ls_infk \
        python src/experiments/semi_supervised_experiment.py  $args  -experiment_key <EXPERIMENT KEY> \
           -experiment_name <EXPERIMENT NAME> \
           -seed <SEED> \
           -tag <TAG> "
    # echo  "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
    #     -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
    #     -R 'select[gpu_model0==$GPU_MODEL]'  \
    #     -G ls_infk \
    #     python src/experiments/semi_supervised_experiment.py  $args -experiment_key 4a9f7ef5de224af5af67ad5b24917bfc \
    #        -experiment_name ssl_hybrid2_512C_CJ_Re128C_Re_Ro  \
    #        -seed  5 \
    #        -tag hyb2_cross -tag no_ytb"
    # echo  "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
    # -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
    # -R 'select[gpu_model0==$GPU_MODEL]'  \
    # -G ls_infk \
    # python src/experiments/semi_supervised_experiment.py  $args -experiment_key 7dfc517b9c744264923469cc65bfb543 \
    #    -experiment_name ssl_hybrid2_512C_CJ_Re_Ro128C_Re_Ro  \
    #    -seed  5 \
    #    -tag hyb2_cross -tag no_ytb"
    ;;
PAIR_ABL_IH)
    meta_file="pair_ablative_ih"
    echo "Launching Pair ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("color_jitter"
        "crop"
        "rotate"
        "random_crop")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 -train_ratio 0.9 -resnet_size 18\
         -sources interhand  -tag pair_abl -tag pmlr  -save_top_k 1  -save_period 1 -resnet_size 18 -num_worker $CORES"
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed1 -seed $seed1"
    done
    # for j in "${augmentations[@]}"; do
    #     echo "$j $seed2"
    #     launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    # done
    ;;
PAIR_ABL_IH_DOWN)
    echo "Launching Pairwise ablative studies"
    meta_file="pair_ablative_ih_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag pair_abl -resnet_size 18 -tag pmlr"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/pair_ablative_ih$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/pair_ablative_ih$seed2
    ;;
*)
    echo "Experiment not recognized!"
    echo "(Run $0 -h for help)"
    exit -1
    ;;
esac

echo "All experiment successfully launched!"
