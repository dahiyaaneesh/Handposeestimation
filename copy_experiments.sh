#!/bin/bash

# VARIABLES
FOLDER="d818d4d306954c1dbcfb24cb30f55c8e"
FROM_CLUSTER=true
META_FILE=false # parameter to decide if txt file is moved from data/models/

# Hidden variables

_help_description="Script to transfer the experiment from LabPC to leonhard and vice a
versa
To transfer from  pc to cluster
bash $0 <experiment_name> --pc 
To trasnfer from cluster to pc:
bash $0 <experiment_name> --cluster
"


# Function to move number of experiments from the cluster.
#
# $1 : Name of meta file located in data/models/
move_from_meta_file(){
    echo "Moving $1 experiments to pc."
    while IFS=',' read -r experiment_name experiment_key
            do
                echo "Transfering $experiment_name ($experiment_key)"
                scp -r  \
                "adahiya@login.leonhard.ethz.ch:/cluster/scratch/adahiya/data/models/master-thesis/$experiment_key" \
                "data/models/master-thesis/$experiment_key"
            
            done < $DATA_PATH/models/$1

}

if [ $# -eq 0 ]; then
    echo "No folder provided!"
elif [[ $1 == "-h"  ]]; then
    echo "$_help_description"
            exit 0
else
    FOLDER="$1"
    shift
    while [ -n "$1" ]; do
        case "$1" in
        --cluster)
            FROM_CLUSTER=true
            shift
            ;;
        --pc)
            FROM_CLUSTER=false
            shift
            ;;
        --from_meta_file)
            META_FILE=true
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

# Main process
if $FROM_CLUSTER eq true; then
    if $META_FILE eq true; then
        move_from_meta_file $FOLDER
    else
        echo "Copying experiment folder from cluster"
        scp -r  \
        "adahiya@login.leonhard.ethz.ch:/cluster/scratch/adahiya/data/models/master-thesis/$FOLDER" \
        "data/models/master-thesis/$FOLDER"
    fi
else
    if $META_FILE eq true; then
        echo "NOT yet implemented"
    else
        echo "Copying experiment folder from Lab PC"
        scp -r  \
        "data/models/master-thesis/$FOLDER" \
        "adahiya@login.leonhard.ethz.ch:/cluster/scratch/adahiya/data/models/master-thesis/$FOLDER"   
    fi
fi
