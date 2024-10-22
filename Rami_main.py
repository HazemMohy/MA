import os
import pandas as pd
import numpy as np
import voxel_statistics as voxel_stats
import instance_statistics as instance_stats
import visuals
from filehandling import read_nifti, write_nifti

def create_overlays(path_gt, path_prediction, path_result, network_name):
    path_out_network = os.path.join(path_result, f"{network_name}_overlay")
    if not os.path.exists(path_out_network):
        os.mkdir(path_out_network)

    for item in os.listdir(path_prediction):
        print(f"Creating overlay for {item}")
        # Read the individual patches
        patch_gt        = read_nifti(os.path.join(path_gt, item))
        patch_prediction = read_nifti(os.path.join(path_prediction, item))
        patch_gt        = np.squeeze(patch_gt)
        patch_prediction= np.squeeze(patch_prediction)
        overlay = visuals.create_overlay(patch_prediction, patch_gt)
        write_nifti(os.path.join(path_out_network, item), overlay)

def create_voxel_scores(path_gt, path_prediction, path_result, network_name):
    """Calculates the voxel scores TP TN FP FN and saves it to a csv
    Args:
        - path_gt           (str)   : Path to ground truth
        - path_prediction   (str)   : Path to network prediction
        - path_result       (str)   : Path where result csv is saved to
        - network_name      (str)   : Network name, will be in csv name
    Returns:
        - df_scores     (pandas.DataFrame)  : DataFrame containing TP TN FP FN for each patch as well as the sum over all patches
    """
    # Final dataframe containing all scores
    columns     = ["Patch", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "Volumetric Similarity", "Dice"]
    df_scores   = pd.DataFrame(columns=columns)

    for item in os.listdir(path_prediction):
        print(f"Reading {item}")
        # Read the individual patches
        patch_gt        = read_nifti(os.path.join(path_gt, item))
        patch_prediction = read_nifti(os.path.join(path_prediction, item))
        patch_gt        = np.squeeze(patch_gt)
        patch_prediction= np.squeeze(patch_prediction)
        patch_gt[patch_gt > 0] = 1

        # Calculate confusion matrix
        tp, tn, fp, fn  = voxel_stats.confusion_matrix(patch_prediction, patch_gt)

        # Calculate scores
        precision, recall, vs, accuracy, f1_dice = voxel_stats.calc_volumetric_score(tp, tn, fp, fn)

        # Save in temporary DF
        df_item         = pd.DataFrame({"Patch":[item],\
                                        "TP":[tp],\
                                        "TN":[tn],\
                                        "FP":[fp],\
                                        "FN":[fn],\
                                        "Precision":[precision],\
                                        "Recall":[recall],\
                                        "Volumetric Similarity":[vs],\
                                        "Dice":[f1_dice],\
                                        })

        # Concat with final DF
        df_scores       = pd.concat([df_scores, df_item])

    # Calculate the sum
    df_scores.loc["sum"] = df_scores.sum()
    df_scores.loc["sum"]["Patch"] = ""
    df_scores.loc["sum"]["Precision"] = ""
    df_scores.loc["sum"]["Recall"] = ""
    df_scores.loc["sum"]["Volumetric Similarity"] = ""
    df_scores.loc["sum"]["Dice"] = ""

    # Save as csv
    path_scores = os.path.join(path_result,f"{network_name}_voxel_scores.csv")
    df_scores.to_csv(path_scores)
    return df_scores

def create_instance_scores(path_gt, path_prediction, path_result, network_name):
    """Calculates the voxel scores TP FP FN and saves it to a csv
    Args:
        - path_gt                   (str)   : Path to ground truth
        - path_prediction           (str)   : Path to network prediction
        - path_result               (str)   : Path where result csv is saved to
        - network_name              (str)   : Network name, will be in csv name
    Returns:
        - df_scores     (pandas.DataFrame)  : DataFrame containing TP FP FN for each patch as well as the sum over all patches
    """
    # Final dataframe containing all scores
    columns     = ["Patch", "TP", "FP", "FN", "Precision", "Recall", "Dice"]
    df_scores   = pd.DataFrame(columns=columns)

    
    for item in os.listdir(path_prediction):
        
        print(f"Reading {item}")
        # Read the individual patches
        patch_gt        = read_nifti(os.path.join(path_gt, item))
        patch_prediction = read_nifti(os.path.join(path_prediction, item))

        patch_gt        = np.squeeze(patch_gt)
        patch_prediction= np.squeeze(patch_prediction)
        patch_gt[patch_gt > 0] = 1

        # Calculate confusion matrix
        tp, fp, fn = instance_stats.confusion_matrix(patch_prediction, patch_gt)

        # Calculate scores
        precision, recall, f1_dice = instance_stats.calc_instance_score(tp, fp, fn)

        # Save in temporary DF
        df_item         = pd.DataFrame({"Patch":[item],\
                                        "TP":[tp],\
                                        "FP":[fp],\
                                        "FN":[fn],\
                                        "Precision":[precision],\
                                        "Recall":[recall],\
                                        "Dice":[f1_dice],\
                                        })

        # Concat with final DF
        df_scores       = pd.concat([df_scores, df_item])

    # Calculate the sum
    df_scores.loc["sum"] = df_scores.sum()
    df_scores.loc["sum"]["Patch"] = ""
    df_scores.loc["sum"]["Precision"] = ""
    df_scores.loc["sum"]["Recall"] = ""
    df_scores.loc["sum"]["Dice"] = ""

    # Save as csv
    path_scores = os.path.join(path_result,f"{network_name}_instance_scores.csv")
    df_scores.to_csv(path_scores)
    return df_scores

# Name of your network
network_name    = "Hazem_Test"

# Paths 
path_gt         = "/lustre/groups/iterm/Hazem/MA/Runs/run_21121056__DiceCELoss_none/Rami_Voxel_gt" # Path to ground truth
path_prediction = "/lustre/groups/iterm/Hazem/MA/Runs/run_28527005__DiceCELoss_None_Exclude_Encoder/Model_outputs_nifti" # Path to prediction
path_result     = "/lustre/groups/iterm/Hazem/MA/Runs/run_28527005__DiceCELoss_None_Exclude_Encoder/Model_outputs_nifti" # Path where result csv is saved to

create_voxel_scores(path_gt, path_prediction, path_result, network_name)

create_instance_scores(path_gt, path_prediction, path_result, network_name)

create_overlays(path_gt, path_prediction, path_result, network_name)
