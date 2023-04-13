import glob
import os
import subprocess
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from torch.utils.data import DataLoader, Dataset

from rasp_model import (
     CavityModel,
     DownstreamModel,
     ResidueEnvironment,
     ResidueEnvironmentsDataset,
)

from helpers import (
     populate_dfs_with_resenvs,
     fermi_transform,
     ds_pred,
)

from visualization import (
     hist_plot,
)

# Main parameters
DEVICE = "cuda"  # "cpu" or "cuda"
NUM_ENSEMBLE = 10

def run():
    # Set start time
    time.perf_counter()

    # Pre-process PDBs
    print("Pre-processing PDBs...")
    pdb_dir = f"{os.path.dirname(sys.path[0])}/data/test/Speedtest/structure/"
    pdb_filenames = sorted(glob.glob(f"{os.path.dirname(sys.path[0])}/data/test/Speedtest/structure/raw/*.pdb"))

    for pdb_filename in pdb_filenames:
        t0 = time.time()

        pdb = pdb_filename[-8:-4]
        print(f"Analyzing PDB: {pdb}!")

        # Pre-process PDBs
        print(f"Pre-processing...")
        parse_script_path = f'{os.path.dirname(sys.path[0])}/src/pdb_parser_scripts/parse_pdbs_pred_single.sh'
        subprocess.call([parse_script_path, pdb_dir, pdb])
        print("Pre-processing finished.")

        t1 = time.time()
        print(f"Time for pre-processing: {t1-t0}")

        ## Pre-process structure data
        # Create temporary residue environment datasets to more easily match ddG data
        pdb_filenames_ds = [f"{os.path.dirname(sys.path[0])}/data/test/Speedtest/structure/parsed/{pdb}_clean_coordinate_features.npz"]
        dataset_structure = ResidueEnvironmentsDataset(pdb_filenames_ds, transformer=None)
        resenv_dataset = {}
        for resenv in dataset_structure:
            key = (
                f"{resenv.pdb_id}{resenv.chain_id}_{resenv.pdb_residue_number}"
                f"{index_to_one(resenv.restype_index)}"
            )
            resenv_dataset[key] = resenv

        # Populate Rosetta dataframes with wt ResidueEnvironment objects
        df_rosetta = pd.read_csv(f"{os.path.dirname(sys.path[0])}/data/test/Speedtest/ddG_Rosetta/ddg.csv")
        df_rosetta = df_rosetta.rename(columns={"score": "score_rosetta"})
        df_rosetta = df_rosetta[df_rosetta["pdbid"]==f"{pdb}"]
        n_rosetta_start = len(df_rosetta)
        
        populate_dfs_with_resenvs(df_rosetta, resenv_dataset)
        print(
            f"{n_rosetta_start-len(df_rosetta)} data points dropped when (inner) matching Rosetta and structure in: Speedtest data set."
        )
        df_total = df_rosetta

        # Do Fermi transform
        df_total["score_rosetta_fermi"] = df_total["score_rosetta"].apply(fermi_transform)    

        # Define models
        best_cavity_model_path = open(f"{os.path.dirname(sys.path[0])}/output/cavity_models/best_model_path.txt", "r").read().strip()
        cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
        cavity_model_net.load_state_dict(torch.load(f"{os.path.dirname(sys.path[0])}/output/cavity_models/{best_cavity_model_path}", map_location=DEVICE))
        cavity_model_net.eval()
        ds_model_net = DownstreamModel().to(DEVICE)
        ds_model_net.eval()

        t2 = time.time()
        print(f"Time for ML data structure initializations: {t2-t1}")

        # Make ML predictions
        print(f"Starting downstream model prediction")
        dataset_key="Speedtest"
        df_ml = ds_pred(cavity_model_net,
                        ds_model_net,
                        df_total,
                        dataset_key,
                        NUM_ENSEMBLE,
                        DEVICE,
                        ) 
        print(f"Finished downstream model prediction")

        t3 = time.time()
        print(f"Time for ML predictions: {t3-t2}")
        print(f"Total time: {t3-t0}")
