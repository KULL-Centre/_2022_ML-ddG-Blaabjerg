#!/bin/bash

# Clean pdbs
dir=$(pwd)
reduce_exe=$dir/pdb_parser_scripts/reduce/reduce_src/reduce
pdb_dir=$1
pdb=$2

# Create data directories
mkdir -p $pdb_dir/cleaned
mkdir -p $pdb_dir/parsed

python $dir/pdb_parser_scripts/clean_pdb.py --pdb_file_in $pdb_dir/raw/$pdb.pdb  \
                                            --out_dir $pdb_dir/cleaned/ \
                                            --reduce_exe $reduce_exe #&> /dev/null

# Check for exit code 0 and skip file if not 0.
if [ $? -eq 0 ]
then
echo "Successfully cleaned $pdb. $counter/$n_pdbs."
else
echo "Error when cleaning $pdb. Skipping.." >&2
fi

# Parse pdbs and save in npz format
pdb_clean="$pdb"_clean

python $dir/pdb_parser_scripts/extract_environments.py --pdb_in $pdb_dir/cleaned/$pdb_clean.pdb  \
                                                       --out_dir $pdb_dir/parsed  &> /dev/null

# Check for exit code 0 and skip file if not 0.
if [ $? -eq 0 ]
then
base_pdb=$(basename $pdb_clean)
echo "Successfully parsed $pdb_clean. \
Finished $counter/$n_pdbs."
else
echo "Error extracting $pdb_clean. Skipping.." >&2
fi
