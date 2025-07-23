#!/bin/bash
# run by calling ./save_runtime.sh <model_name> <module>
# outputs the total runtime in the file "<module>/models/<model>/runtime.txt"

model="$1"
module="$2"

sum=$(cat $module/models/$model/output.o | grep -E "\s*Run time \s*:\s*([0-9]*)\s.*" | grep -Eo "[0-9]*" | awk '{total += $1} END {print total}')

output_filepath="$module/models/$model/runtime.txt"
echo "$sum" > "$output_filepath"

echo "Sum of run times saved to '$output_filepath', and got $sum"
