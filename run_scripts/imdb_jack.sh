#!/bin/bash

set -e  # Exit on any error

# COMMAND TO RUN:
# ./run6.sh > run6.log 2>&1 &


# Available datasets
datasets=("dblp" "gith" "imdb" "uspt")
dblp=${datasets[0]}
gith=${datasets[1]}
imdb=${datasets[2]}
uspt=${datasets[3]}

# Available dataset paths
dataset_paths=("../data/preprocessed/dblp/toy.dblp.v12.json" "../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3" "../data/preprocessed/gith/gith.data.csv.filtered.mt75.ts3" "../data/preprocessed/imdb/imdb.title.basics.tsv.filtered.mt75.ts3" "../data/preprocessed/uspt/uspt.patent.tsv.filtered.mt75.ts3")
dblp_toy_path=${dataset_paths[0]}
dblp_path=${dataset_paths[1]}
gith_path=${dataset_paths[2]}
imdb_path=${dataset_paths[3]}
uspt_path=${dataset_paths[4]}

# ------------------------------------------------------------------------------
# CONFIGURATIONS
# ------------------------------------------------------------------------------


script_name=$(basename "$0" .sh)

# get first item split by -
model=$(echo $script_name | cut -d "-" -f 1)

# get second item split by -

# gpus to use
gpus="6,7"

# Select dataset
dataset=$imdb
dataset_path=$imdb_path
is_toy=false


# ------------------------------------------------------------------------------
# END CONFIGURATIONS
# ------------------------------------------------------------------------------


# Log file for all run times
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
times_dir="${SCRIPT_DIR}/../run_times"
logs_dir="${SCRIPT_DIR}/../run_logs"

# Ensure the directories exist
mkdir -p ../run_logs

cd ../src

echo ""
echo "Processing model ${script_name}."

# Get the start time
start_time=$(date +%s)

if [ $is_toy = true ]; then
  is_toy_msg="toy_"
else
  is_toy_msg=""
fi


# Run the command with nohup and capture its PID
echo "Running: nohup python3 -u main.py ..."
nohup python3 -u main.py \
  -data $dataset_path \
  -domain $dataset \
  -model "nmt_${script_name}" \
  -gpus $gpus \
  > "../run_logs/${is_toy_msg}${script_name}.log" 2> "../run_logs/${is_toy_msg}${script_name}_errors.log" &

pid=$!

echo ""
echo "Started process $pid for model: ${script_name}"
echo ""

# Wait for the process to complete
wait $pid

# Get the end time
end_time=$(date +%s)

# Calculate the elapsed time in seconds
elapsed_time=$(($end_time - $start_time))

# Convert elapsed time into hours, minutes, and seconds
hours=$(($elapsed_time / 3600))
minutes=$(($elapsed_time % 3600 / 60))
seconds=$(($elapsed_time % 60))

# Format the elapsed time as Xh Xm Xs
formatted_time="${hours}h ${minutes}m ${seconds}s"

in_minutes=$(($elapsed_time / 60))

echo "Process completed for model: ${script_name}."
echo "Duration: ${formatted_time} (${in_minutes} mins)."
echo ""