###
import xgboost as xgb
import matplotlib.pyplot as plt
import uproot as uproot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import shutil
from array import array
import math
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [16, 8]
#import os
import ROOT
print(ROOT.gROOT.GetVersion())
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendTextSize(0.035)
ROOT.gStyle.SetLegendFont(62)
ROOT.gStyle.SetLabelFont(62)
import os
from math import sqrt, isnan
from scipy.stats import chi2

import matplotlib as mpl
import matplotlib.lines as mlines
mpl.rcParams['savefig.dpi'] = 300
#mpl.rcParams['mathtext.fallback_to_cm'] = False
#mpl.rcParams['font.size'] = 10
mpl.rcParams['figure.titlesize'] = 'xx-large'
mpl.rcParams['axes.labelsize'] = 'xx-large'
mpl.rcParams['xtick.labelsize'] = 'xx-large'
mpl.rcParams['ytick.labelsize'] = 'xx-large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['legend.title_fontsize'] = 'larger'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['hatch.linewidth'] = 0.5
title_size = 24
label_size = 0.035

import warnings
warnings.filterwarnings('ignore')
#import os
import itertools

#Category number labels
# -100 = lee, -1=cc 1g overlay, -2=nc del 1g overlay, -3=nc pi0 overlay, 
# 111 = outFV sig, 0=cc1g, 1=nc other 1g, 2=nc del 1g, 3=nc pi0 1g, 
# 4=nue, 5=nc bkd, 6=ncpi0 bkd, 7=numu bkd, 8=numupi0 bkd, 9=nfv, 10=cosmic, 11=dirt, 12=extbnb, 13=data

def LoadTreesTruth(file1, file2, file3):
    with uproot.open(file1)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_1 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_1 = f_in_pfeval_over.arrays(pfeval_variables + truth_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_1 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_1 = f_in_eval_over.arrays(eval_variables + eval_truth_variables, library="pd")


    with uproot.open(file2)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_2 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_2 = f_in_pfeval_over.arrays(pfeval_variables + truth_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_2 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_2 = f_in_eval_over.arrays(eval_variables + eval_truth_variables, library="pd")


    with uproot.open(file3)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_3 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_3 = f_in_pfeval_over.arrays(pfeval_variables + truth_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_3 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_3 = f_in_eval_over.arrays(eval_variables + eval_truth_variables, library="pd")

    all_df_in_bdt_over = pd.concat([all_df_in_bdt_over_1, all_df_in_bdt_over_2, all_df_in_bdt_over_3], ignore_index=True, sort=False)

    del all_df_in_bdt_over_1
    del all_df_in_bdt_over_2
    del all_df_in_bdt_over_3

    all_df_in_pfeval_over = pd.concat([all_df_in_pfeval_over_1, all_df_in_pfeval_over_2, all_df_in_pfeval_over_3], ignore_index=True, sort=False)

    del all_df_in_pfeval_over_1
    del all_df_in_pfeval_over_2
    del all_df_in_pfeval_over_3

    all_df_in_kine_over = pd.concat([all_df_in_kine_over_1, all_df_in_kine_over_2, all_df_in_kine_over_3], ignore_index=True, sort=False)

    del all_df_in_kine_over_1
    del all_df_in_kine_over_2
    del all_df_in_kine_over_3

    all_df_in_eval_over = pd.concat([all_df_in_eval_over_1, all_df_in_eval_over_2, all_df_in_eval_over_3], ignore_index=True, sort=False)

    del all_df_in_eval_over_1
    del all_df_in_eval_over_2
    del all_df_in_eval_over_3

    return all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over

###
def LoadTreesTruth1(file1):
    with uproot.open(file1)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over = f_in_pfeval_over.arrays(pfeval_variables + truth_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over = f_in_eval_over.arrays(eval_variables + eval_truth_variables, library="pd")

    return all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over

###
def LoadTreesData(file1, file2, file3):
    with uproot.open(file1)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_1 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_1 = f_in_pfeval_over.arrays(pfeval_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_1 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_1 = f_in_eval_over.arrays(eval_variables, library="pd")


    with uproot.open(file2)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_2 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_2 = f_in_pfeval_over.arrays(pfeval_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_2 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file2)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_2 = f_in_eval_over.arrays(eval_variables, library="pd")


    with uproot.open(file3)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over_3 = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over_3 = f_in_pfeval_over.arrays(pfeval_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over_3 = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file3)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over_3 = f_in_eval_over.arrays(eval_variables, library="pd")

    all_df_in_bdt_over = pd.concat([all_df_in_bdt_over_1, all_df_in_bdt_over_2, all_df_in_bdt_over_3], ignore_index=True, sort=False)

    del all_df_in_bdt_over_1
    del all_df_in_bdt_over_2
    del all_df_in_bdt_over_3

    all_df_in_pfeval_over = pd.concat([all_df_in_pfeval_over_1, all_df_in_pfeval_over_2, all_df_in_pfeval_over_3], ignore_index=True, sort=False)

    del all_df_in_pfeval_over_1
    del all_df_in_pfeval_over_2
    del all_df_in_pfeval_over_3

    all_df_in_kine_over = pd.concat([all_df_in_kine_over_1, all_df_in_kine_over_2, all_df_in_kine_over_3], ignore_index=True, sort=False)

    del all_df_in_kine_over_1
    del all_df_in_kine_over_2
    del all_df_in_kine_over_3

    all_df_in_eval_over = pd.concat([all_df_in_eval_over_1, all_df_in_eval_over_2, all_df_in_eval_over_3], ignore_index=True, sort=False)

    del all_df_in_eval_over_1
    del all_df_in_eval_over_2
    del all_df_in_eval_over_3

    return all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over

###
def LoadTreesData1(file1):
    with uproot.open(file1)["wcpselection/T_BDTvars"] as f_in_bdt_over:
        all_df_in_bdt_over = f_in_bdt_over.arrays(bdt_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_PFeval"] as f_in_pfeval_over:
        all_df_in_pfeval_over = f_in_pfeval_over.arrays(pfeval_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_KINEvars"] as f_in_kine_over:
        all_df_in_kine_over = f_in_kine_over.arrays(kine_variables, library="pd")

    with uproot.open(file1)["wcpselection/T_eval"] as f_in_eval_over:
        all_df_in_eval_over = f_in_eval_over.arrays(eval_variables, library="pd")

    return all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over

###
def LoadBNBOverlay(all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over):
    #bnb overlay
    true_event_types = []
    shw_sp_energy = []
    single_photon_numu_score = []
    single_photon_other_score = []
    single_photon_ncpi0_score = []
    single_photon_nue_score = []
    weight_cv = all_df_in_eval_over["weight_cv"].to_numpy()
    weight_spline = all_df_in_eval_over["weight_spline"].to_numpy()
    is_sigoverlay_vec = []

    kine_reco_Enu_vec = all_df_in_kine_over["kine_reco_Enu"].to_numpy()
    shw_sp_n_20mev_showers_vec = all_df_in_bdt_over["shw_sp_n_20mev_showers"].to_numpy()
    reco_nuvtxX_vec = all_df_in_pfeval_over["reco_nuvtxX"].to_numpy()
    truth_muonMomentum = all_df_in_pfeval_over["truth_muonMomentum"].to_numpy()

    reco_nuvtxY_vec = all_df_in_pfeval_over["reco_nuvtxY"].to_numpy()
    reco_nuvtxZ_vec = all_df_in_pfeval_over["reco_nuvtxZ"].to_numpy()
    reco_showervtxX_vec = all_df_in_pfeval_over["reco_showervtxX"].to_numpy()
    reco_showervtxY_vec = all_df_in_pfeval_over["reco_showervtxY"].to_numpy()
    reco_showervtxZ_vec = all_df_in_pfeval_over["reco_showervtxZ"].to_numpy()
    reco_showerMomentum_vec = all_df_in_pfeval_over["reco_showerMomentum"].to_numpy()
    reco_showerMomentum0_vec = [] 
    reco_showerMomentum1_vec = [] 
    reco_showerMomentum2_vec = [] 
    reco_showerMomentum3_vec = [] 
    truth_showerMomentum_vec = all_df_in_pfeval_over["truth_showerMomentum"].to_numpy()
    truth_showerMomentum0_vec = [] 
    truth_showerMomentum1_vec = [] 
    truth_showerMomentum2_vec = [] 
    truth_showerMomentum3_vec = []
    #reco_muonMomentum3_vec = all_df_in_pfeval_over["reco_muonMomentum[3]"].to_numpy()
    reco_muonMomentum_vec = all_df_in_pfeval_over["reco_muonMomentum"].to_numpy()
    match_energy_vec = all_df_in_eval_over["match_energy"].to_numpy()
    truth_nuEnergy_vec = all_df_in_eval_over["truth_nuEnergy"].to_numpy()
    truth_energyInside_vec = all_df_in_eval_over["truth_energyInside"].to_numpy()
    truth_showerKE_vec = all_df_in_pfeval_over["truth_showerKE"].to_numpy()
    
    N_protons = []
    true_N_protons = []
    kine_energy_particle_vec = all_df_in_kine_over["kine_energy_particle"].to_numpy()
    kine_particle_type_vec = all_df_in_kine_over["kine_particle_type"].to_numpy()

    r = all_df_in_pfeval_over["run"].to_numpy()
    s = all_df_in_pfeval_over["subrun"].to_numpy()
    e = all_df_in_pfeval_over["event"].to_numpy()

    time = []
    #evtTimeNS_vec = all_df_in_time_over["evtTimeNS_cor"].to_numpy()

    for i in range(len(kine_reco_Enu_vec)):
        #if (e[i] != pnd_evt[i]):
        #    print("Event number mismatch between wc and pelee: ", i, e[i], pnd_evt[i])
        is_sigoverlay_vec.append(0)
        time.append(-9999.0)
        event_time = -9999.0
        isSig = False
        isNC1g = False
        isCC1g = False
        truth_muonEnergy = truth_muonMomentum[i][3] - 0.105658
        match_completeness_energy = (all_df_in_eval_over["match_completeness_energy"].to_numpy())[i]
        truth_energyInside = (all_df_in_eval_over["truth_energyInside"].to_numpy())[i]
        truthSinglePhoton = (all_df_in_pfeval_over["truth_single_photon"].to_numpy())[i]
        truthisCC = (all_df_in_pfeval_over["truth_isCC"].to_numpy())[i]
        truth_NCDelta = (all_df_in_pfeval_over["truth_NCDelta"].to_numpy())[i]
        truth_showerMother = (all_df_in_pfeval_over["truth_showerMother"].to_numpy())[i]
        truth_nuPdg = (all_df_in_eval_over["truth_nuPdg"].to_numpy())[i]
        truth_vtxInside = (all_df_in_eval_over["truth_vtxInside"].to_numpy())[i]
        truth_Npi0 = (all_df_in_pfeval_over["truth_Npi0"].to_numpy())[i]
        truth_showerMomentum0_vec.append(truth_showerMomentum_vec[i][0])
        truth_showerMomentum1_vec.append(truth_showerMomentum_vec[i][1])
        truth_showerMomentum2_vec.append(truth_showerMomentum_vec[i][2])
        truth_showerMomentum3_vec.append(truth_showerMomentum_vec[i][3])
        reco_showerMomentum0_vec.append(reco_showerMomentum_vec[i][0])
        reco_showerMomentum1_vec.append(reco_showerMomentum_vec[i][1])
        reco_showerMomentum2_vec.append(reco_showerMomentum_vec[i][2])
        reco_showerMomentum3_vec.append(reco_showerMomentum_vec[i][3])
        if ((match_completeness_energy/truth_energyInside)>0.1 and (truthSinglePhoton==1 )) :
            if (not truthisCC):
                    isNC1g = True
            if (truthisCC and abs(truth_nuPdg)==14 and truth_muonEnergy<0.1) :
                    isCC1g = True
                    if(truth_vtxInside):
                        true_event_types.append(0)
                    else:
                        true_event_types.append(111)
        if (isNC1g or isCC1g):
            isSig = True
            ##num_sig+=1
        if (isNC1g):
            if(not truth_vtxInside):
                true_event_types.append(111)
            else:
                if (truth_NCDelta==1):
                    true_event_types.append(2)
                elif (truth_showerMother==111):
                    true_event_types.append(3)
                else:
                    true_event_types.append(1)

        if (not isSig):
            #num_bkg+=1
            if (truth_energyInside!=0 and (match_completeness_energy/truth_energyInside)>0.1):
                if (truthisCC and abs(truth_nuPdg) == 14 and truth_vtxInside):
                    if (truth_Npi0>0):
                        true_event_types.append(8)
                    else:
                        true_event_types.append(7)
                elif (not truthisCC and truth_vtxInside==1):
                    if (truth_Npi0>0):
                        true_event_types.append(6)
                    else:
                        true_event_types.append(5)
                if (truthisCC and abs(truth_nuPdg) == 12 and truth_vtxInside):
                    true_event_types.append(4)
                if (not truth_vtxInside): 
                    true_event_types.append(9)
            else:
                true_event_types.append(10)
        
        #num_true_protons = 0
        #for j in range(all_df_in_pfeval_over["truth_Ntrack"].to_numpy()[i]):
        #        pdgcode = (all_df_in_pfeval_over["truth_pdg"].to_numpy()[i])[j]
        #        mother = (all_df_in_pfeval_over["truth_mother"].to_numpy()[i])[j]
        #        energy = (all_df_in_pfeval_over["truth_startMomentum"].to_numpy()[i])[j][3]
        #        if(abs(pdgcode)==2212 and mother==0 and energy - 0.938272 > 0.035):
        #            num_true_protons += 1;
        #true_N_protons.append(num_true_protons)

        #for getting eff/pur of preselection
        #if (kine_reco_Enu_vec[i] >= 0):
            #num_tot_gen+=1
         #   if (isSig):
                #num_sig_gen+=1
          #  if (reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
                #num_tot_x_vertex+=1
           #     if (isSig):
                    #num_sig_x_vertex+=1
        
        #Merge Peaks
        #gap=18.936
        #Shift=7292.0
        #TThelp=0.0         
        #TThelp=evtTimeNS_vec[i]-Shift+gap*0.5
        #TT_merged = -9999.
        ##merge peaks
        #if(TThelp>=0. and TThelp<gap*81.0):
        #    TT_merged=(TThelp-int(TThelp/gap)*gap)-gap*0.5 
        #event_time = TT_merged
        #time.append(event_time)
    #
        #event_time_pnd = -9999.0
        #TT_help_pnd = pnd_time_vec[i]-Shift+gap*0.5
        #TT_merged_pnd = -9999.
        ##merge peaks
        #if(TT_help_pnd>=0. and TT_help_pnd<gap*81.0):
        #    TT_merged_pnd=(TT_help_pnd-int(TT_help_pnd/gap)*gap)-gap*0.5
        #event_time_pnd = TT_merged_pnd
        #pnd_time.append(event_time_pnd)

        shw_sp_energy.append((all_df_in_bdt_over["shw_sp_energy"].to_numpy())[i])

        num_protons = 0
        kine_energy_particle = np.array(kine_energy_particle_vec[i]) 
        kine_particle_type = kine_particle_type_vec[i]
        proton_mask = (np.abs(kine_particle_type) == 2212) & (kine_energy_particle > 35)
        num_protons = np.sum(proton_mask)
        N_protons.append(num_protons)
        
        if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0): 
            if (math.isnan(all_df_in_bdt_over["single_photon_numu_score"].to_numpy()[i])):
                single_photon_numu_score.append(-99999.0)
            else:
                single_photon_numu_score.append((all_df_in_bdt_over["single_photon_numu_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_other_score"].to_numpy()[i])):
                single_photon_other_score.append(-99999.0)
            else:
                single_photon_other_score.append((all_df_in_bdt_over["single_photon_other_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_ncpi0_score"].to_numpy()[i])):
                single_photon_ncpi0_score.append(-99999.0)
            else:
                single_photon_ncpi0_score.append((all_df_in_bdt_over["single_photon_ncpi0_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_nue_score"].to_numpy()[i])):
                single_photon_nue_score.append(-99999.0)
            else:
                single_photon_nue_score.append((all_df_in_bdt_over["single_photon_nue_score"].to_numpy())[i])
        else:
            #shw_sp_energy.append(-99999.0)
            single_photon_numu_score.append(-99999.0)
            single_photon_other_score.append(-99999.0)
            single_photon_ncpi0_score.append(-99999.0)
            single_photon_nue_score.append(-99999.0)
            #N_protons.append(-1)

        
        


    all_df_in_bdt_over["true_event_type"] = true_event_types
    all_df_in_bdt_over["shw_sp_energy"] = shw_sp_energy
    #all_df_in_bdt_over["kine_reco_Enu"] = kine_reco_Enu_vec
    all_df_in_bdt_over["single_photon_numu_score"] = single_photon_numu_score
    all_df_in_bdt_over["single_photon_other_score"] = single_photon_other_score
    all_df_in_bdt_over["single_photon_ncpi0_score"] = single_photon_ncpi0_score
    all_df_in_bdt_over["single_photon_nue_score"] = single_photon_nue_score
    all_df_in_bdt_over["weight_cv"] = weight_cv
    all_df_in_bdt_over["weight_spline"] = weight_spline
    all_df_in_bdt_over["reco_nuvtxX"] = reco_nuvtxX_vec
    all_df_in_bdt_over["reco_nuvtxY"] = reco_nuvtxY_vec
    all_df_in_bdt_over["reco_nuvtxZ"] = reco_nuvtxZ_vec
    all_df_in_bdt_over["reco_showervtxX"] = reco_showervtxX_vec
    all_df_in_bdt_over["reco_showervtxY"] = reco_showervtxY_vec
    all_df_in_bdt_over["reco_showervtxZ"] = reco_showervtxZ_vec
    all_df_in_bdt_over["reco_showerMomentum0"] = reco_showerMomentum0_vec
    all_df_in_bdt_over["reco_showerMomentum1"] = reco_showerMomentum1_vec
    all_df_in_bdt_over["reco_showerMomentum2"] = reco_showerMomentum2_vec
    all_df_in_bdt_over["reco_showerMomentum3"] = reco_showerMomentum3_vec
    all_df_in_bdt_over["truth_showerMomentum0"] = truth_showerMomentum0_vec
    all_df_in_bdt_over["truth_showerMomentum1"] = truth_showerMomentum1_vec
    all_df_in_bdt_over["truth_showerMomentum2"] = truth_showerMomentum2_vec
    all_df_in_bdt_over["truth_showerMomentum3"] = truth_showerMomentum3_vec
    #all_df_in_bdt_over["reco_muonMomentum"] = reco_muonMomentum_vec
    all_df_in_bdt_over["is_sigoverlay"] = is_sigoverlay_vec
    all_df_in_bdt_over["match_energy"] = match_energy_vec
    all_df_in_bdt_over["truth_nuEnergy"] = truth_nuEnergy_vec
    all_df_in_bdt_over["truth_energyInside"] = truth_energyInside_vec
    all_df_in_bdt_over["truth_showerKE"] = truth_showerKE_vec
    all_df_in_bdt_over["N_protons"] = N_protons
    all_df_in_bdt_over["run"] = r
    all_df_in_bdt_over["subrun"] = s
    all_df_in_bdt_over["event"] = e
    all_df_in_bdt_over["time"] = time
    #all_df_in_bdt_over["pnd_time"] = pnd_time

    all_df_in_bdt_over = all_df_in_bdt_over.join(all_df_in_kine_over)
    #all_df_in_bdt_over = all_df_in_bdt_over.join(all_df_in_time_over)

    return all_df_in_bdt_over

###
def LoadDirt(all_df_in_bdt_dirt, all_df_in_pfeval_dirt, all_df_in_kine_dirt, all_df_in_eval_dirt):
    #dirt
    true_event_types = []
    shw_sp_energy = []
    single_photon_numu_score = []
    single_photon_other_score = []
    single_photon_ncpi0_score = []
    single_photon_nue_score = []
    weight_cv = all_df_in_eval_dirt["weight_cv"].to_numpy()
    weight_spline = all_df_in_eval_dirt["weight_spline"].to_numpy()
    is_sigoverlay_vec = []

    kine_reco_Enu_vec = all_df_in_kine_dirt["kine_reco_Enu"].to_numpy()
    shw_sp_n_20mev_showers_vec = all_df_in_bdt_dirt["shw_sp_n_20mev_showers"].to_numpy()
    reco_nuvtxX_vec = all_df_in_pfeval_dirt["reco_nuvtxX"].to_numpy()

    reco_nuvtxY_vec = all_df_in_pfeval_dirt["reco_nuvtxY"].to_numpy()
    reco_nuvtxZ_vec = all_df_in_pfeval_dirt["reco_nuvtxZ"].to_numpy()
    reco_showervtxX_vec = all_df_in_pfeval_dirt["reco_showervtxX"].to_numpy()
    reco_showervtxY_vec = all_df_in_pfeval_dirt["reco_showervtxY"].to_numpy()
    reco_showervtxZ_vec = all_df_in_pfeval_dirt["reco_showervtxZ"].to_numpy()
    reco_showerMomentum_vec = all_df_in_pfeval_dirt["reco_showerMomentum"].to_numpy()
    reco_showerMomentum0_vec = [] 
    reco_showerMomentum1_vec = [] 
    reco_showerMomentum2_vec = [] 
    reco_showerMomentum3_vec = [] 
    truth_showerMomentum_vec = all_df_in_pfeval_dirt["truth_showerMomentum"].to_numpy()
    truth_showerMomentum0_vec = [] 
    truth_showerMomentum1_vec = [] 
    truth_showerMomentum2_vec = [] 
    truth_showerMomentum3_vec = []
    reco_muonMomentum_vec = all_df_in_pfeval_dirt["reco_muonMomentum"].to_numpy()
    truth_muonMomentum = all_df_in_pfeval_dirt["truth_muonMomentum"].to_numpy()
    match_energy_vec = all_df_in_eval_dirt["match_energy"].to_numpy()
    truth_nuEnergy_vec = all_df_in_eval_dirt["truth_nuEnergy"].to_numpy()
    truth_energyInside_vec = all_df_in_eval_dirt["truth_energyInside"].to_numpy()
    truth_showerKE_vec = all_df_in_pfeval_dirt["truth_showerKE"].to_numpy()
    truth_nuPdg = (all_df_in_eval_dirt["truth_nuPdg"].to_numpy())
    N_protons = []
    kine_energy_particle_vec = all_df_in_kine_dirt["kine_energy_particle"].to_numpy()
    kine_particle_type_vec = all_df_in_kine_dirt["kine_particle_type"].to_numpy()

    r = all_df_in_pfeval_dirt["run"].to_numpy()
    s = all_df_in_pfeval_dirt["subrun"].to_numpy()
    e = all_df_in_pfeval_dirt["event"].to_numpy()

    time = []
    #evtTimeNS_vec = all_df_in_time_dirt["evtTimeNS_cor"].to_numpy()
    #pnd_time = []
    #pnd_time_vec = all_df_in_pelee_dirt["interaction_time_abs"].to_numpy()

    #pnd_evt = all_df_in_pelee_dirt["evt"].to_numpy()

    for i in range(len(kine_reco_Enu_vec)):
        #if (e[i] != pnd_evt[i]):
        #    print("Event number mismatch between wc and pelee: ", i, e[i], pnd_evt[i])
        is_sigoverlay_vec.append(0)
        time.append(-9999.0)
        event_time = -9999.0
        isSig = False
        isNC1g = False
        isCC1g = False
        truth_muonEnergy = (truth_muonMomentum[i][3]) - 0.105658
        match_completeness_energy = (all_df_in_eval_dirt["match_completeness_energy"].to_numpy())[i]
        truth_energyInside = (all_df_in_eval_dirt["truth_energyInside"].to_numpy())[i]
        truthSinglePhoton = (all_df_in_pfeval_dirt["truth_single_photon"].to_numpy())[i]
        truthisCC = (all_df_in_pfeval_dirt["truth_isCC"].to_numpy())[i]
        truth_vtxInside = (all_df_in_eval_dirt["truth_vtxInside"].to_numpy())[i]
        truth_showerMomentum0_vec.append(truth_showerMomentum_vec[i][0])
        truth_showerMomentum1_vec.append(truth_showerMomentum_vec[i][1])
        truth_showerMomentum2_vec.append(truth_showerMomentum_vec[i][2])
        truth_showerMomentum3_vec.append(truth_showerMomentum_vec[i][3])
        reco_showerMomentum0_vec.append(reco_showerMomentum_vec[i][0])
        reco_showerMomentum1_vec.append(reco_showerMomentum_vec[i][1])
        reco_showerMomentum2_vec.append(reco_showerMomentum_vec[i][2])
        reco_showerMomentum3_vec.append(reco_showerMomentum_vec[i][3])
        if ((match_completeness_energy/truth_energyInside)>0.1 and (truthSinglePhoton==1 )) :
            if (not truthisCC):
                    isNC1g = True
            if (truthisCC and abs(truth_nuPdg[i])==14 and truth_muonEnergy<0.1) :
                    isCC1g = True

        if (isNC1g or isCC1g):
            isSig = True
            #num_sig+=1
        if (isSig):
            true_event_types.append(111)
        else:
            #num_bkg+=1
            true_event_types.append(11)
        
        #for getting eff/pur of preselection
        #if (kine_reco_Enu_vec[i] >= 0):
            #num_tot_gen+=1
         #   if (isSig):
                #num_sig_gen+=1
          #  if (reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
                #num_tot_x_vertex+=1
           #     if (isSig):
                    #num_sig_x_vertex+=1
        
        ##Merge Peaks
        #gap=18.936
        #Shift=7292.0
        #TThelp=0.0         
        #TThelp=evtTimeNS_vec[i]-Shift+gap*0.5
        #TT_merged = -9999.
        ##merge peaks
        #if(TThelp>=0. and TThelp<gap*81.0):
        #    TT_merged=(TThelp-int(TThelp/gap)*gap)-gap*0.5 
        #event_time = TT_merged
        #time.append(event_time)
    #
        #event_time_pnd = -9999.0
        #TT_help_pnd = pnd_time_vec[i]-Shift+gap*0.5
        #TT_merged_pnd = -9999.
        ##merge peaks
        #if(TT_help_pnd>=0. and TT_help_pnd<gap*81.0):
        #    TT_merged_pnd=(TT_help_pnd-int(TT_help_pnd/gap)*gap)-gap*0.5
        #event_time_pnd = TT_merged_pnd
        #pnd_time.append(event_time_pnd)

        shw_sp_energy.append((all_df_in_bdt_dirt["shw_sp_energy"].to_numpy())[i])

        num_protons = 0
        kine_energy_particle = np.array(kine_energy_particle_vec[i]) 
        kine_particle_type = kine_particle_type_vec[i]
        proton_mask = (np.abs(kine_particle_type) == 2212) & (kine_energy_particle > 35)
        num_protons = np.sum(proton_mask)
        N_protons.append(num_protons)
        
        if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
            if (math.isnan(all_df_in_bdt_dirt["single_photon_numu_score"].to_numpy()[i])):
                single_photon_numu_score.append(-99999.0)
            else:
                single_photon_numu_score.append((all_df_in_bdt_dirt["single_photon_numu_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_dirt["single_photon_other_score"].to_numpy()[i])):
                single_photon_other_score.append(-99999.0)
            else:
                single_photon_other_score.append((all_df_in_bdt_dirt["single_photon_other_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_dirt["single_photon_ncpi0_score"].to_numpy()[i])):
                single_photon_ncpi0_score.append(-99999.0)
            else:
                single_photon_ncpi0_score.append((all_df_in_bdt_dirt["single_photon_ncpi0_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_dirt["single_photon_nue_score"].to_numpy()[i])):
                single_photon_nue_score.append(-99999.0)
            else:
                single_photon_nue_score.append((all_df_in_bdt_dirt["single_photon_nue_score"].to_numpy())[i])
        else:
            #shw_sp_energy.append(-99999.0)
            single_photon_numu_score.append(-99999.0)
            single_photon_other_score.append(-99999.0)
            single_photon_ncpi0_score.append(-99999.0)
            single_photon_nue_score.append(-99999.0)
            #N_protons.append(-1)

    

    all_df_in_bdt_dirt["true_event_type"] = true_event_types
    all_df_in_bdt_dirt["shw_sp_energy"] = shw_sp_energy
    #all_df_in_bdt_dirt["kine_reco_Enu"] = kine_reco_Enu_vec
    all_df_in_bdt_dirt["single_photon_numu_score"] = single_photon_numu_score
    all_df_in_bdt_dirt["single_photon_other_score"] = single_photon_other_score
    all_df_in_bdt_dirt["single_photon_ncpi0_score"] = single_photon_ncpi0_score
    all_df_in_bdt_dirt["single_photon_nue_score"] = single_photon_nue_score
    all_df_in_bdt_dirt["weight_cv"] = weight_cv
    all_df_in_bdt_dirt["weight_spline"] = weight_spline
    all_df_in_bdt_dirt["reco_nuvtxX"] = reco_nuvtxX_vec
    all_df_in_bdt_dirt["reco_nuvtxY"] = reco_nuvtxY_vec
    all_df_in_bdt_dirt["reco_nuvtxZ"] = reco_nuvtxZ_vec
    all_df_in_bdt_dirt["reco_showervtxX"] = reco_showervtxX_vec
    all_df_in_bdt_dirt["reco_showervtxY"] = reco_showervtxY_vec
    all_df_in_bdt_dirt["reco_showervtxZ"] = reco_showervtxZ_vec
    all_df_in_bdt_dirt["reco_showerMomentum0"] = reco_showerMomentum0_vec
    all_df_in_bdt_dirt["reco_showerMomentum1"] = reco_showerMomentum1_vec
    all_df_in_bdt_dirt["reco_showerMomentum2"] = reco_showerMomentum2_vec
    all_df_in_bdt_dirt["reco_showerMomentum3"] = reco_showerMomentum3_vec
    #all_df_in_bdt_dirt["reco_muonMomentum"] = reco_muonMomentum_vec
    all_df_in_bdt_dirt["is_sigoverlay"] = is_sigoverlay_vec
    all_df_in_bdt_dirt["match_energy"] = match_energy_vec
    all_df_in_bdt_dirt["truth_nuEnergy"] = truth_nuEnergy_vec
    all_df_in_bdt_dirt["truth_energyInside"] = truth_energyInside_vec
    all_df_in_bdt_dirt["truth_showerKE"] = truth_showerKE_vec
    all_df_in_bdt_dirt["truth_showerMomentum0"] = truth_showerMomentum0_vec
    all_df_in_bdt_dirt["truth_showerMomentum1"] = truth_showerMomentum1_vec
    all_df_in_bdt_dirt["truth_showerMomentum2"] = truth_showerMomentum2_vec
    all_df_in_bdt_dirt["truth_showerMomentum3"] = truth_showerMomentum3_vec
    all_df_in_bdt_dirt["N_protons"] = N_protons
    all_df_in_bdt_dirt["run"] = r
    all_df_in_bdt_dirt["subrun"] = s
    all_df_in_bdt_dirt["event"] = e
    all_df_in_bdt_dirt["time"] = time
    #all_df_in_bdt_dirt["pnd_time"] = pnd_time

    all_df_in_bdt_dirt = all_df_in_bdt_dirt.join(all_df_in_kine_dirt)
    #all_df_in_bdt_dirt = all_df_in_bdt_dirt.join(all_df_in_time_dirt)

    return all_df_in_bdt_dirt

###
def LoadExtBnb(all_df_in_bdt_ext, all_df_in_pfeval_ext, all_df_in_kine_ext, all_df_in_eval_ext):
    #extbnb
    true_event_types = []
    shw_sp_energy = []
    single_photon_numu_score = []
    single_photon_other_score = []
    single_photon_ncpi0_score = []
    single_photon_nue_score = []
    weight_cv = []
    weight_spline = []
    is_sigoverlay_vec = []
    truth_nuEnergy_vec = []
    truth_energyInside_vec = []
    truth_showerKE_vec = []
    truth_showerMomentum0_vec = []
    truth_showerMomentum1_vec = []
    truth_showerMomentum2_vec = []
    truth_showerMomentum3_vec = []
    kine_reco_Enu_vec = all_df_in_kine_ext["kine_reco_Enu"].to_numpy()
    shw_sp_n_20mev_showers_vec = all_df_in_bdt_ext["shw_sp_n_20mev_showers"].to_numpy()
    reco_nuvtxX_vec = all_df_in_pfeval_ext["reco_nuvtxX"].to_numpy()

    reco_nuvtxY_vec = all_df_in_pfeval_ext["reco_nuvtxY"].to_numpy()
    reco_nuvtxZ_vec = all_df_in_pfeval_ext["reco_nuvtxZ"].to_numpy()
    reco_showervtxX_vec = all_df_in_pfeval_ext["reco_showervtxX"].to_numpy()
    reco_showervtxY_vec = all_df_in_pfeval_ext["reco_showervtxY"].to_numpy()
    reco_showervtxZ_vec = all_df_in_pfeval_ext["reco_showervtxZ"].to_numpy()
    reco_showerMomentum_vec = all_df_in_pfeval_ext["reco_showerMomentum"].to_numpy()
    reco_showerMomentum0_vec = [] 
    reco_showerMomentum1_vec = [] 
    reco_showerMomentum2_vec = [] 
    reco_showerMomentum3_vec = [] 
    reco_muonMomentum_vec = all_df_in_pfeval_ext["reco_muonMomentum"].to_numpy()
    match_energy_vec = all_df_in_eval_ext["match_energy"].to_numpy()
    N_protons = []
    kine_energy_particle_vec = all_df_in_kine_ext["kine_energy_particle"].to_numpy()
    kine_particle_type_vec = all_df_in_kine_ext["kine_particle_type"].to_numpy()

    r = all_df_in_pfeval_ext["run"].to_numpy()
    s = all_df_in_pfeval_ext["subrun"].to_numpy()
    e = all_df_in_pfeval_ext["event"].to_numpy()

    time = []
    #evtTimeNS_vec = all_df_in_time_ext["evtTimeNS"].to_numpy()
    #pnd_time = []
    #pnd_time_vec = all_df_in_pelee_ext["interaction_time_abs"].to_numpy()

    #pnd_evt = all_df_in_pelee_ext["evt"].to_numpy()

    for i in range(len(kine_reco_Enu_vec)):
        #if (e[i] != pnd_evt[i]):
        #    print("Event number mismatch between wc and pelee: ", i, e[i], pnd_evt[i])
        #num_bkg+=1
        true_event_types.append(12)
        weight_cv.append(1.0)
        weight_spline.append(1.0)
        is_sigoverlay_vec.append(0)
        truth_nuEnergy_vec.append(-1.0)
        truth_energyInside_vec.append(-1.0)
        truth_showerKE_vec.append(-1.0)
        truth_showerMomentum0_vec.append(-1.0)
        truth_showerMomentum1_vec.append(-1.0)
        truth_showerMomentum2_vec.append(-1.0)
        truth_showerMomentum3_vec.append(-1.0)
        reco_showerMomentum0_vec.append(reco_showerMomentum_vec[i][0])
        reco_showerMomentum1_vec.append(reco_showerMomentum_vec[i][1])
        reco_showerMomentum2_vec.append(reco_showerMomentum_vec[i][2])
        reco_showerMomentum3_vec.append(reco_showerMomentum_vec[i][3])
        time.append(-9999.0)
        event_time = -9999.0
        
        #for getting eff/pur of preselection
       # if (kine_reco_Enu_vec[i] >= 0):
            #num_tot_gen+=1
        #    if (reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
                #num_tot_x_vertex+=1
        
    # #Merge Peaks
    # gap=18.936
    # Shift=7292.0
    # TThelp=0.0         
    # TThelp=evtTimeNS_vec[i]-Shift+gap*0.5
    # TT_merged = -9999.
    # #merge peaks
    # if(TThelp>=0. and TThelp<gap*81.0):
    #     TT_merged=(TThelp-int(TThelp/gap)*gap)-gap*0.5 
    # event_time = TT_merged
    # time.append(event_time)

    # event_time_pnd = -9999.0
    # TT_help_pnd = pnd_time_vec[i]-Shift+gap*0.5
    # TT_merged_pnd = -9999.
    # #merge peaks
    # if(TT_help_pnd>=0. and TT_help_pnd<gap*81.0):
    #     TT_merged_pnd=(TT_help_pnd-int(TT_help_pnd/gap)*gap)-gap*0.5
    # event_time_pnd = TT_merged_pnd
    # pnd_time.append(event_time_pnd)

        shw_sp_energy.append(0.95*(all_df_in_bdt_ext["shw_sp_energy"].to_numpy())[i])

        num_protons = 0
        kine_energy_particle = np.array(kine_energy_particle_vec[i]) 
        kine_particle_type = kine_particle_type_vec[i]
        proton_mask = (np.abs(kine_particle_type) == 2212) & (kine_energy_particle > 35)
        num_protons = np.sum(proton_mask)
        N_protons.append(num_protons)
        
        if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
            if (math.isnan(all_df_in_bdt_ext["single_photon_numu_score"].to_numpy()[i])):
                single_photon_numu_score.append(-99999.0)
            else:
                single_photon_numu_score.append((all_df_in_bdt_ext["single_photon_numu_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_ext["single_photon_other_score"].to_numpy()[i])):
                single_photon_other_score.append(-99999.0)
            else:
                single_photon_other_score.append((all_df_in_bdt_ext["single_photon_other_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_ext["single_photon_ncpi0_score"].to_numpy()[i])):
                single_photon_ncpi0_score.append(-99999.0)
            else:
                single_photon_ncpi0_score.append((all_df_in_bdt_ext["single_photon_ncpi0_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_ext["single_photon_nue_score"].to_numpy()[i])):
                single_photon_nue_score.append(-99999.0)
            else:
                single_photon_nue_score.append((all_df_in_bdt_ext["single_photon_nue_score"].to_numpy())[i])
        else:
            #shw_sp_energy.append(-99999.0)
            single_photon_numu_score.append(-99999.0)
            single_photon_other_score.append(-99999.0)
            single_photon_ncpi0_score.append(-99999.0)
            single_photon_nue_score.append(-99999.0)
            #N_protons.append(-1)

        


    all_df_in_bdt_ext["true_event_type"] = true_event_types
    all_df_in_bdt_ext["shw_sp_energy"] = shw_sp_energy
    #all_df_in_bdt_ext["kine_reco_Enu"] = kine_reco_Enu_vec
    all_df_in_bdt_ext["single_photon_numu_score"] = single_photon_numu_score
    all_df_in_bdt_ext["single_photon_other_score"] = single_photon_other_score
    all_df_in_bdt_ext["single_photon_ncpi0_score"] = single_photon_ncpi0_score
    all_df_in_bdt_ext["single_photon_nue_score"] = single_photon_nue_score
    all_df_in_bdt_ext["weight_cv"] = weight_cv
    all_df_in_bdt_ext["weight_spline"] = weight_spline
    all_df_in_bdt_ext["reco_nuvtxX"] = reco_nuvtxX_vec
    all_df_in_bdt_ext["reco_nuvtxY"] = reco_nuvtxY_vec
    all_df_in_bdt_ext["reco_nuvtxZ"] = reco_nuvtxZ_vec
    all_df_in_bdt_ext["reco_showervtxX"] = reco_showervtxX_vec
    all_df_in_bdt_ext["reco_showervtxY"] = reco_showervtxY_vec
    all_df_in_bdt_ext["reco_showervtxZ"] = reco_showervtxZ_vec
    all_df_in_bdt_ext["reco_showerMomentum0"] = reco_showerMomentum0_vec
    all_df_in_bdt_ext["reco_showerMomentum1"] = reco_showerMomentum1_vec
    all_df_in_bdt_ext["reco_showerMomentum2"] = reco_showerMomentum2_vec
    all_df_in_bdt_ext["reco_showerMomentum3"] = reco_showerMomentum3_vec
    #all_df_in_bdt_ext["reco_muonMomentum"] = reco_muonMomentum_vec
    all_df_in_bdt_ext["is_sigoverlay"] = is_sigoverlay_vec
    all_df_in_bdt_ext["match_energy"] = match_energy_vec
    all_df_in_bdt_ext["truth_nuEnergy"] = truth_nuEnergy_vec
    all_df_in_bdt_ext["truth_energyInside"] = truth_energyInside_vec
    all_df_in_bdt_ext["truth_showerKE"] = truth_showerKE_vec
    all_df_in_bdt_ext["truth_showerMomentum0"] = truth_showerMomentum0_vec
    all_df_in_bdt_ext["truth_showerMomentum1"] = truth_showerMomentum1_vec
    all_df_in_bdt_ext["truth_showerMomentum2"] = truth_showerMomentum2_vec
    all_df_in_bdt_ext["truth_showerMomentum3"] = truth_showerMomentum3_vec
    all_df_in_bdt_ext["N_protons"] = N_protons
    all_df_in_bdt_ext["run"] = r
    all_df_in_bdt_ext["subrun"] = s
    all_df_in_bdt_ext["event"] = e
    all_df_in_bdt_ext["time"] = time
    #all_df_in_bdt_ext["pnd_time"] = pnd_time

    all_df_in_bdt_ext = all_df_in_bdt_ext.join(all_df_in_kine_ext)
    #all_df_in_bdt_ext = all_df_in_bdt_ext.join(all_df_in_time_ext)

    return all_df_in_bdt_ext

###
def LoadBnb(all_df_in_bdt_data, all_df_in_pfeval_data, all_df_in_kine_data, all_df_in_eval_data):
    #bnb data
    true_event_types = []
    shw_sp_energy = []
    single_photon_numu_score = []
    single_photon_other_score = []
    single_photon_ncpi0_score = []
    single_photon_nue_score = []
    weight_cv = []
    weight_spline = []
    is_sigoverlay_vec = []
    truth_nuEnergy_vec = []
    truth_energyInside_vec = []
    truth_showerKE_vec = []
    truth_showerMomentum0_vec = []
    truth_showerMomentum1_vec = []
    truth_showerMomentum2_vec = []
    truth_showerMomentum3_vec = []

    kine_reco_Enu_vec = all_df_in_kine_data["kine_reco_Enu"].to_numpy()
    shw_sp_n_20mev_showers_vec = all_df_in_bdt_data["shw_sp_n_20mev_showers"].to_numpy()
    reco_nuvtxX_vec = all_df_in_pfeval_data["reco_nuvtxX"].to_numpy()

    reco_nuvtxY_vec = all_df_in_pfeval_data["reco_nuvtxY"].to_numpy()
    reco_nuvtxZ_vec = all_df_in_pfeval_data["reco_nuvtxZ"].to_numpy()
    reco_showervtxX_vec = all_df_in_pfeval_data["reco_showervtxX"].to_numpy()
    reco_showervtxY_vec = all_df_in_pfeval_data["reco_showervtxY"].to_numpy()
    reco_showervtxZ_vec = all_df_in_pfeval_data["reco_showervtxZ"].to_numpy()
    reco_showerMomentum_vec = all_df_in_pfeval_data["reco_showerMomentum"].to_numpy()
    reco_showerMomentum0_vec = [] 
    reco_showerMomentum1_vec = [] 
    reco_showerMomentum2_vec = [] 
    reco_showerMomentum3_vec = [] 
    reco_muonMomentum_vec = all_df_in_pfeval_data["reco_muonMomentum"].to_numpy()
    match_energy_vec = all_df_in_eval_data["match_energy"].to_numpy()
    N_protons = []
    kine_energy_particle_vec = all_df_in_kine_data["kine_energy_particle"].to_numpy()
    kine_particle_type_vec = all_df_in_kine_data["kine_particle_type"].to_numpy()

    r_data = all_df_in_pfeval_data["run"].to_numpy()
    s_data = all_df_in_pfeval_data["subrun"].to_numpy()
    e_data = all_df_in_pfeval_data["event"].to_numpy()

    evtTimeNS_vec = [] #all_df_in_time_data["evtTimeNS"].to_numpy()
    time = []
    #pnd_time = []
    #pnd_time_vec = all_df_in_pelee_data["interaction_time_abs"].to_numpy()

    for i in range(len(kine_reco_Enu_vec)):
        #if (e[i] != pnd_evt[i]):
        #    print("Event number mismatch between wc and pelee: ", i, e[i], pnd_evt[i])
        true_event_types.append(13)
        weight_cv.append(1.0)
        weight_spline.append(1.0)
        is_sigoverlay_vec.append(0)
        truth_nuEnergy_vec.append(-1.0)
        truth_energyInside_vec.append(-1.0)
        truth_showerKE_vec.append(-1.0)
        truth_showerMomentum0_vec.append(-1.0)
        truth_showerMomentum1_vec.append(-1.0)
        truth_showerMomentum2_vec.append(-1.0)
        truth_showerMomentum3_vec.append(-1.0)
        reco_showerMomentum0_vec.append(reco_showerMomentum_vec[i][0])
        reco_showerMomentum1_vec.append(reco_showerMomentum_vec[i][1])
        reco_showerMomentum2_vec.append(reco_showerMomentum_vec[i][2])
        reco_showerMomentum3_vec.append(reco_showerMomentum_vec[i][3])
        event_time = -9999.
        
        
        #if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
        #Merge Peaks
        gap=18.936
        Shift=0.0
        TThelp=0.0
        if (r_data[i] >= 19500):
            Shift=2920.5
        elif (r_data[i] >= 17380): 
            Shift=2916.0
        elif (r_data[i] >= 13697):
            Shift = 3147.3 #3166.1
        elif (r_data[i] >= 10812):
            Shift = 3568.5
        elif (r_data[i] >= 8321):
            Shift = 3610.7
        elif (r_data[i] >= 5800): 
            Shift = 3164.4
        elif (r_data[i] >= 0):
            Shift = 3168.9         
        TThelp= -9999.#evtTimeNS_vec[i]-Shift+gap*0.5
        TT_merged = -9999.
        #merge peaks
        if(TThelp>=0. and TThelp<gap*81.0):
            TT_merged=(TThelp-int(TThelp/gap)*gap)-gap*0.5 
        event_time = TT_merged
        
        #time.append(event_time)

        time.append(-1)

        #event_time_pnd = -9999.0
        #TT_help_pnd = pnd_time_vec[i]-Shift+gap*0.5
        #TT_merged_pnd = -9999.
        #merge peaks
        #if(TT_help_pnd>=0. and TT_help_pnd<gap*81.0):
        #    TT_merged_pnd=(TT_help_pnd-int(TT_help_pnd/gap)*gap)-gap*0.5
        #event_time_pnd = TT_merged_pnd
        #pnd_time.append(event_time_pnd)

        shw_sp_energy.append(0.95*(all_df_in_bdt_data["shw_sp_energy"].to_numpy())[i])

        num_protons = 0
        kine_energy_particle = np.array(kine_energy_particle_vec[i]) 
        kine_particle_type = kine_particle_type_vec[i]
        proton_mask = (np.abs(kine_particle_type) == 2212) & (kine_energy_particle > 35)
        num_protons = np.sum(proton_mask)
        N_protons.append(num_protons)
            
        if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):# and event_time > -6.6 and event_time < 3.4):
            if (math.isnan(all_df_in_bdt_data["single_photon_numu_score"].to_numpy()[i])):
                single_photon_numu_score.append(-99999.0)
            else:
                single_photon_numu_score.append((all_df_in_bdt_data["single_photon_numu_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_data["single_photon_other_score"].to_numpy()[i])):
                single_photon_other_score.append(-99999.0)
            else:
                single_photon_other_score.append((all_df_in_bdt_data["single_photon_other_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_data["single_photon_ncpi0_score"].to_numpy()[i])):
                single_photon_ncpi0_score.append(-99999.0)
            else:
                single_photon_ncpi0_score.append((all_df_in_bdt_data["single_photon_ncpi0_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_data["single_photon_nue_score"].to_numpy()[i])):
                single_photon_nue_score.append(-99999.0)
            else:
                single_photon_nue_score.append((all_df_in_bdt_data["single_photon_nue_score"].to_numpy())[i])
        else:
            #shw_sp_energy.append(-99999.0)
            single_photon_numu_score.append(-99999.0)
            single_photon_other_score.append(-99999.0)
            single_photon_ncpi0_score.append(-99999.0)
            single_photon_nue_score.append(-99999.0)
            #N_protons.append(-1)


    all_df_in_bdt_data["true_event_type"] = true_event_types
    all_df_in_bdt_data["shw_sp_energy"] = shw_sp_energy
    #all_df_in_bdt_data["kine_reco_Enu"] = kine_reco_Enu_vec
    all_df_in_bdt_data["single_photon_numu_score"] = single_photon_numu_score
    all_df_in_bdt_data["single_photon_other_score"] = single_photon_other_score
    all_df_in_bdt_data["single_photon_ncpi0_score"] = single_photon_ncpi0_score
    all_df_in_bdt_data["single_photon_nue_score"] = single_photon_nue_score
    all_df_in_bdt_data["weight_cv"] = weight_cv
    all_df_in_bdt_data["weight_spline"] = weight_spline
    all_df_in_bdt_data["reco_nuvtxX"] = reco_nuvtxX_vec
    all_df_in_bdt_data["reco_nuvtxY"] = reco_nuvtxY_vec
    all_df_in_bdt_data["reco_nuvtxZ"] = reco_nuvtxZ_vec
    all_df_in_bdt_data["reco_showervtxX"] = reco_showervtxX_vec
    all_df_in_bdt_data["reco_showervtxY"] = reco_showervtxY_vec
    all_df_in_bdt_data["reco_showervtxZ"] = reco_showervtxZ_vec
    all_df_in_bdt_data["reco_showerMomentum0"] = reco_showerMomentum0_vec
    all_df_in_bdt_data["reco_showerMomentum1"] = reco_showerMomentum1_vec
    all_df_in_bdt_data["reco_showerMomentum2"] = reco_showerMomentum2_vec
    all_df_in_bdt_data["reco_showerMomentum3"] = reco_showerMomentum3_vec
    #all_df_in_bdt_data["reco_muonMomentum"] = reco_muonMomentum_vec
    all_df_in_bdt_data["is_sigoverlay"] = is_sigoverlay_vec
    all_df_in_bdt_data["match_energy"] = match_energy_vec
    all_df_in_bdt_data["truth_nuEnergy"] = truth_nuEnergy_vec
    all_df_in_bdt_data["truth_energyInside"] = truth_energyInside_vec
    all_df_in_bdt_data["truth_showerKE"] = truth_showerKE_vec
    all_df_in_bdt_data["truth_showerMomentum0"] = truth_showerMomentum0_vec
    all_df_in_bdt_data["truth_showerMomentum1"] = truth_showerMomentum1_vec
    all_df_in_bdt_data["truth_showerMomentum2"] = truth_showerMomentum2_vec
    all_df_in_bdt_data["truth_showerMomentum3"] = truth_showerMomentum3_vec
    all_df_in_bdt_data["N_protons"] = N_protons
    all_df_in_bdt_data["run"] = r_data
    all_df_in_bdt_data["subrun"] = s_data
    all_df_in_bdt_data["event"] = e_data
    all_df_in_bdt_data["time"] = time
    #all_df_in_bdt_data["pnd_time"] = pnd_time

    all_df_in_bdt_data = all_df_in_bdt_data.join(all_df_in_kine_data)
    #all_df_in_bdt_data = all_df_in_bdt_data.join(all_df_in_time_data)

    return all_df_in_bdt_data

###
def LoadNCPi0Overlay(all_df_in_bdt_over, all_df_in_pfeval_over, all_df_in_kine_over, all_df_in_eval_over):
    #ncpi0 overlay overlay
    true_event_types = []
    true_event_types_sub = []
    shw_sp_energy = []
    single_photon_numu_score = []
    single_photon_other_score = []
    single_photon_ncpi0_score = []
    single_photon_nue_score = []
    weight_cv = all_df_in_eval_over["weight_cv"].to_numpy()
    weight_spline = all_df_in_eval_over["weight_spline"].to_numpy()
    is_sigoverlay_vec = []

    kine_reco_Enu_vec = all_df_in_kine_over["kine_reco_Enu"].to_numpy()
    shw_sp_n_20mev_showers_vec = all_df_in_bdt_over["shw_sp_n_20mev_showers"].to_numpy()
    reco_nuvtxX_vec = all_df_in_pfeval_over["reco_nuvtxX"].to_numpy()
    truth_muonMomentum = all_df_in_pfeval_over["truth_muonMomentum"].to_numpy()

    reco_nuvtxY_vec = all_df_in_pfeval_over["reco_nuvtxY"].to_numpy()
    reco_nuvtxZ_vec = all_df_in_pfeval_over["reco_nuvtxZ"].to_numpy()
    reco_showervtxX_vec = all_df_in_pfeval_over["reco_showervtxX"].to_numpy()
    reco_showervtxY_vec = all_df_in_pfeval_over["reco_showervtxY"].to_numpy()
    reco_showervtxZ_vec = all_df_in_pfeval_over["reco_showervtxZ"].to_numpy()
    reco_showerMomentum_vec = all_df_in_pfeval_over["reco_showerMomentum"].to_numpy()
    reco_showerMomentum0_vec = [] 
    reco_showerMomentum1_vec = [] 
    reco_showerMomentum2_vec = [] 
    reco_showerMomentum3_vec = [] 
    truth_showerMomentum_vec = all_df_in_pfeval_over["truth_showerMomentum"].to_numpy()
    truth_showerMomentum0_vec = [] 
    truth_showerMomentum1_vec = [] 
    truth_showerMomentum2_vec = [] 
    truth_showerMomentum3_vec = []
    #reco_muonMomentum3_vec = all_df_in_pfeval_over["reco_muonMomentum[3]"].to_numpy()
    reco_muonMomentum_vec = all_df_in_pfeval_over["reco_muonMomentum"].to_numpy()
    match_energy_vec = all_df_in_eval_over["match_energy"].to_numpy()
    truth_nuEnergy_vec = all_df_in_eval_over["truth_nuEnergy"].to_numpy()
    truth_energyInside_vec = all_df_in_eval_over["truth_energyInside"].to_numpy()
    truth_showerKE_vec = all_df_in_pfeval_over["truth_showerKE"].to_numpy()
    
    N_protons = []
    true_N_protons = []
    kine_energy_particle_vec = all_df_in_kine_over["kine_energy_particle"].to_numpy()
    kine_particle_type_vec = all_df_in_kine_over["kine_particle_type"].to_numpy()

    r = all_df_in_pfeval_over["run"].to_numpy()
    s = all_df_in_pfeval_over["subrun"].to_numpy()
    e = all_df_in_pfeval_over["event"].to_numpy()

    time = []
    #evtTimeNS_vec = all_df_in_time_over["evtTimeNS_cor"].to_numpy()

    for i in range(len(kine_reco_Enu_vec)):
        #if (e[i] != pnd_evt[i]):
        #    print("Event number mismatch between wc and pelee: ", i, e[i], pnd_evt[i])
        true_event_types.append(-3)
        is_sigoverlay_vec.append(0)
        time.append(-9999.0)
        event_time = -9999.0
        isSig = False
        isNC1g = False
        isCC1g = False
        truth_muonEnergy = truth_muonMomentum[i][3] - 0.105658
        match_completeness_energy = (all_df_in_eval_over["match_completeness_energy"].to_numpy())[i]
        truth_energyInside = (all_df_in_eval_over["truth_energyInside"].to_numpy())[i]
        truthSinglePhoton = (all_df_in_pfeval_over["truth_single_photon"].to_numpy())[i]
        truthisCC = (all_df_in_pfeval_over["truth_isCC"].to_numpy())[i]
        truth_NCDelta = (all_df_in_pfeval_over["truth_NCDelta"].to_numpy())[i]
        truth_showerMother = (all_df_in_pfeval_over["truth_showerMother"].to_numpy())[i]
        truth_nuPdg = (all_df_in_eval_over["truth_nuPdg"].to_numpy())[i]
        truth_vtxInside = (all_df_in_eval_over["truth_vtxInside"].to_numpy())[i]
        truth_Npi0 = (all_df_in_pfeval_over["truth_Npi0"].to_numpy())[i]
        truth_showerMomentum0_vec.append(truth_showerMomentum_vec[i][0])
        truth_showerMomentum1_vec.append(truth_showerMomentum_vec[i][1])
        truth_showerMomentum2_vec.append(truth_showerMomentum_vec[i][2])
        truth_showerMomentum3_vec.append(truth_showerMomentum_vec[i][3])
        reco_showerMomentum0_vec.append(reco_showerMomentum_vec[i][0])
        reco_showerMomentum1_vec.append(reco_showerMomentum_vec[i][1])
        reco_showerMomentum2_vec.append(reco_showerMomentum_vec[i][2])
        reco_showerMomentum3_vec.append(reco_showerMomentum_vec[i][3])
        if ((match_completeness_energy/truth_energyInside)>0.1 and (truthSinglePhoton==1 )) :
            if (not truthisCC):
                    isNC1g = True
            if (truthisCC and abs(truth_nuPdg)==14 and truth_muonEnergy<0.1) :
                    isCC1g = True
                    if(truth_vtxInside):
                        true_event_types_sub.append(0)
                    else:
                        true_event_types_sub.append(111)
        if (isNC1g or isCC1g):
            isSig = True
            ##num_sig+=1
        if (isNC1g):
            if(not truth_vtxInside):
                true_event_types_sub.append(111)
            else:
                if (truth_NCDelta==1):
                    true_event_types_sub.append(2)
                elif (truth_showerMother==111):
                    true_event_types_sub.append(3)
                else:
                    true_event_types_sub.append(1)

        if (not isSig):
            #num_bkg+=1
            if (truth_energyInside!=0 and (match_completeness_energy/truth_energyInside)>0.1):
                if (truthisCC and abs(truth_nuPdg) == 14 and truth_vtxInside):
                    if (truth_Npi0>0):
                        true_event_types_sub.append(8)
                    else:
                        true_event_types_sub.append(7)
                elif (not truthisCC and truth_vtxInside==1):
                    if (truth_Npi0>0):
                        true_event_types_sub.append(6)
                    else:
                        true_event_types_sub.append(5)
                if (truthisCC and abs(truth_nuPdg) == 12 and truth_vtxInside):
                    true_event_types_sub.append(4)
                if (not truth_vtxInside): 
                    true_event_types_sub.append(9)
            else:
                true_event_types_sub.append(10)

        #num_true_protons = 0
        #for j in range(all_df_in_pfeval_over["truth_Ntrack"].to_numpy()[i]):
        #        pdgcode = (all_df_in_pfeval_over["truth_pdg"].to_numpy()[i])[j]
        #        mother = (all_df_in_pfeval_over["truth_mother"].to_numpy()[i])[j]
        #        energy = (all_df_in_pfeval_over["truth_startMomentum"].to_numpy()[i])[j][3]
        #        if(abs(pdgcode)==2212 and mother==0 and energy - 0.938272 > 0.035):
        #            num_true_protons += 1;
        #true_N_protons.append(num_true_protons)

        #for getting eff/pur of preselection
        #if (kine_reco_Enu_vec[i] >= 0):
            #num_tot_gen+=1
         #   if (isSig):
                #num_sig_gen+=1
          #  if (reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0):
                #num_tot_x_vertex+=1
           #     if (isSig):
                    #num_sig_x_vertex+=1
        
        #Merge Peaks
        #gap=18.936
        #Shift=7292.0
        #TThelp=0.0         
        #TThelp=evtTimeNS_vec[i]-Shift+gap*0.5
        #TT_merged = -9999.
        ##merge peaks
        #if(TThelp>=0. and TThelp<gap*81.0):
        #    TT_merged=(TThelp-int(TThelp/gap)*gap)-gap*0.5 
        #event_time = TT_merged
        #time.append(event_time)
    #
        #event_time_pnd = -9999.0
        #TT_help_pnd = pnd_time_vec[i]-Shift+gap*0.5
        #TT_merged_pnd = -9999.
        ##merge peaks
        #if(TT_help_pnd>=0. and TT_help_pnd<gap*81.0):
        #    TT_merged_pnd=(TT_help_pnd-int(TT_help_pnd/gap)*gap)-gap*0.5
        #event_time_pnd = TT_merged_pnd
        #pnd_time.append(event_time_pnd)

        shw_sp_energy.append((all_df_in_bdt_over["shw_sp_energy"].to_numpy())[i])

        num_protons = 0
        kine_energy_particle = np.array(kine_energy_particle_vec[i]) 
        kine_particle_type = kine_particle_type_vec[i]
        proton_mask = (np.abs(kine_particle_type) == 2212) & (kine_energy_particle > 35)
        num_protons = np.sum(proton_mask)
        N_protons.append(num_protons)
        
        if (kine_reco_Enu_vec[i] >= 0 and shw_sp_n_20mev_showers_vec[i]>0 and reco_nuvtxX_vec[i]>5.0 and reco_nuvtxX_vec[i]<250.0): 
            if (math.isnan(all_df_in_bdt_over["single_photon_numu_score"].to_numpy()[i])):
                single_photon_numu_score.append(-99999.0)
            else:
                single_photon_numu_score.append((all_df_in_bdt_over["single_photon_numu_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_other_score"].to_numpy()[i])):
                single_photon_other_score.append(-99999.0)
            else:
                single_photon_other_score.append((all_df_in_bdt_over["single_photon_other_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_ncpi0_score"].to_numpy()[i])):
                single_photon_ncpi0_score.append(-99999.0)
            else:
                single_photon_ncpi0_score.append((all_df_in_bdt_over["single_photon_ncpi0_score"].to_numpy())[i])
            if (math.isnan(all_df_in_bdt_over["single_photon_nue_score"].to_numpy()[i])):
                single_photon_nue_score.append(-99999.0)
            else:
                single_photon_nue_score.append((all_df_in_bdt_over["single_photon_nue_score"].to_numpy())[i])
        else:
            #shw_sp_energy.append(-99999.0)
            single_photon_numu_score.append(-99999.0)
            single_photon_other_score.append(-99999.0)
            single_photon_ncpi0_score.append(-99999.0)
            single_photon_nue_score.append(-99999.0)
            #N_protons.append(-1)

        
        


    all_df_in_bdt_over["true_event_type"] = true_event_types
    all_df_in_bdt_over["true_event_type_sub"] = true_event_types_sub
    all_df_in_bdt_over["shw_sp_energy"] = shw_sp_energy
    #all_df_in_bdt_over["kine_reco_Enu"] = kine_reco_Enu_vec
    all_df_in_bdt_over["single_photon_numu_score"] = single_photon_numu_score
    all_df_in_bdt_over["single_photon_other_score"] = single_photon_other_score
    all_df_in_bdt_over["single_photon_ncpi0_score"] = single_photon_ncpi0_score
    all_df_in_bdt_over["single_photon_nue_score"] = single_photon_nue_score
    all_df_in_bdt_over["weight_cv"] = weight_cv
    all_df_in_bdt_over["weight_spline"] = weight_spline
    all_df_in_bdt_over["reco_nuvtxX"] = reco_nuvtxX_vec
    all_df_in_bdt_over["reco_nuvtxY"] = reco_nuvtxY_vec
    all_df_in_bdt_over["reco_nuvtxZ"] = reco_nuvtxZ_vec
    all_df_in_bdt_over["reco_showervtxX"] = reco_showervtxX_vec
    all_df_in_bdt_over["reco_showervtxY"] = reco_showervtxY_vec
    all_df_in_bdt_over["reco_showervtxZ"] = reco_showervtxZ_vec
    all_df_in_bdt_over["reco_showerMomentum0"] = reco_showerMomentum0_vec
    all_df_in_bdt_over["reco_showerMomentum1"] = reco_showerMomentum1_vec
    all_df_in_bdt_over["reco_showerMomentum2"] = reco_showerMomentum2_vec
    all_df_in_bdt_over["reco_showerMomentum3"] = reco_showerMomentum3_vec
    all_df_in_bdt_over["truth_showerMomentum0"] = truth_showerMomentum0_vec
    all_df_in_bdt_over["truth_showerMomentum1"] = truth_showerMomentum1_vec
    all_df_in_bdt_over["truth_showerMomentum2"] = truth_showerMomentum2_vec
    all_df_in_bdt_over["truth_showerMomentum3"] = truth_showerMomentum3_vec
    #all_df_in_bdt_over["reco_muonMomentum"] = reco_muonMomentum_vec
    all_df_in_bdt_over["is_sigoverlay"] = is_sigoverlay_vec
    all_df_in_bdt_over["match_energy"] = match_energy_vec
    all_df_in_bdt_over["truth_nuEnergy"] = truth_nuEnergy_vec
    all_df_in_bdt_over["truth_energyInside"] = truth_energyInside_vec
    all_df_in_bdt_over["truth_showerKE"] = truth_showerKE_vec
    all_df_in_bdt_over["N_protons"] = N_protons
    all_df_in_bdt_over["run"] = r
    all_df_in_bdt_over["subrun"] = s
    all_df_in_bdt_over["event"] = e
    all_df_in_bdt_over["time"] = time
    #all_df_in_bdt_over["pnd_time"] = pnd_time

    all_df_in_bdt_over = all_df_in_bdt_over.join(all_df_in_kine_over)
    #all_df_in_bdt_over = all_df_in_bdt_over.join(all_df_in_time_over)

    return all_df_in_bdt_over

###
def GetVariableArrays(all_df, var, array_name, array_sig = [0,1,2,3,111], selection = "all", ignore_cat = []):
    #read in the variable with name var from the root file and make it into sig, bkg, and data arrays called
    #[array_name]_sig, [array_name]_bkg, [array_name]_data
    #var: string of variable name in root file
    #array_name: name of variable array

    single_photon_numu_score = all_df["single_photon_numu_score"].to_numpy()
    single_photon_other_score = all_df["single_photon_other_score"].to_numpy()
    single_photon_ncpi0_score = all_df["single_photon_ncpi0_score"].to_numpy()
    single_photon_nue_score = all_df["single_photon_nue_score"].to_numpy()
    num_shw = all_df["shw_sp_n_20mev_showers"].to_numpy()
    num_pro = all_df["N_protons"].to_numpy()
    r = all_df["run"].to_numpy()
    s = all_df["subrun"].to_numpy()
    e = all_df["event"].to_numpy()
    
    var_array = all_df[var].to_numpy()
    
    var_array_sig = []
    var_array_bkg = []
    var_array_data = []

    y = all_df["true_event_type"].to_numpy()
    num_evts = all_df.shape[0]

    for i in range(num_evts):
        if y[i] not in ignore_cat and PassSelection(selection, single_photon_numu_score[i], single_photon_other_score[i], single_photon_ncpi0_score[i], single_photon_nue_score[i], num_shw[i], num_pro[i], r[i], s[i], e[i]):
            if y[i] in array_sig:
                var_array_sig.append(var_array[i])
            elif y[i] == 13:
                var_array_data.append(var_array[i])
            elif y[i] > -1: #y[i]>3 and y[i]!=13 and y[i]<100:
                var_array_bkg.append(var_array[i])
    
    array_name_sig = array_name+"_sig"
    array_name_bkg = array_name+"_bkg"
    array_name_data = array_name+"_data"
    globals()[array_name_sig] = var_array_sig
    globals()[array_name_bkg] = var_array_bkg
    globals()[array_name_data] = var_array_data
    
    return var_array_sig, var_array_bkg, var_array_data

###
def GetPOT(file):
    # for calculating the POT of a file
    #returns p: the pot of the file

    #f_in = uproot.open(file)["wcpselection/T_pot"]
    subrun = 0
    pot_vars = ["pot_tor875good","subRunNo"]
    with uproot.open(file)["wcpselection/T_pot"] as f_in:
        all_df_in_fin = f_in.arrays(pot_vars, library="pd")
    pot_tor875 = all_df_in_fin["pot_tor875good"].to_numpy()
    p = np.sum(pot_tor875) 
    
    print(p)
            
    return p

###
def CalculateWeights(all_df, run1dataPOT, run2dataPOT, run3dataPOT, run1ExtBnbPOT, run2ExtBnbPOT, run3ExtBnbPOT, pot_vars):
    # for calculating the weight
    #returns w: array filled with weights

    for var_name, file_name in pot_vars:
        globals()[var_name] = GetPOT(file_name)
    
    weight_cv = all_df["weight_cv"].to_numpy()
    weight_spline = all_df["weight_spline"].to_numpy()
    is_ext = (all_df["true_event_type"].to_numpy() == 12) #% 10 == 4)
    is_mccosmic = (all_df["true_event_type"].to_numpy() == 10)
    is_dirt = (all_df["true_event_type"].to_numpy() == 11)
    is_data = (all_df["true_event_type"].to_numpy() == 13)
    is_lee = (all_df["true_event_type"].to_numpy() == -100)
    is_sigoverlay = (all_df["is_sigoverlay"].to_numpy() == 1)
    is_ncpi0overlay = (all_df["true_event_type"].to_numpy() == -3)
    has_muon = (all_df["is_sigoverlay"].to_numpy() == 0)# (all_df["reco_muonMomentum"].to_numpy() > 0)
    POT_factor = [(run1dataPOT + run2dataPOT + run3dataPOT) / (run1ExtBnbPOT + run2ExtBnbPOT + run3ExtBnbPOT) if is_ext[i] else
                  (run1dataPOT + run2dataPOT + run3dataPOT) / (run1DirtPOT + run2DirtPOT + run3DirtPOT) if is_dirt[i] else
                  (run1dataPOT + run2dataPOT + run3dataPOT) / (run1SPPOT + run2SPPOT + run3SPPOT) if is_sigoverlay[i] else
                  1. if is_data[i] else
                  1. if is_lee[i] else
                  1. if is_ncpi0overlay[i] else
                  (run1dataPOT + run2dataPOT + run3dataPOT) / (run1BnbPOT + run2BnbPOT + run3BnbPOT) for i in range(len(is_ext))]
    #[1.0 for i in range(len(is_ext)) ]
    
    #POT_factor = [ 5e19 / (run1ExtBnbPOT + run3ExtBnbPOT) if is_ext[i] else
    #              5e19 / (run1dataPOT + run3dataPOT) if is_data[i] else 
    #              5e19 / (run1BnbPOT + run3BnbPOT) for i in range(len(is_ext))]
    #POT_factor = [(run1BnbPOT + run3BnbPOT)/(run1ExtBnbPOT + run3ExtBnbPOT) if is_ext[i] 
    #                    else 1. for i in range(len(is_ext))]
    
    #nueff_mu = ((run1frac*0.766) + (run2frac*0.828) + (run3frac*0.797)) / (run1frac + run2frac + run3frac)
    #nueff_nomu = ((run1frac*0.78) + (run2frac*0.86) + (run3frac*0.82)) / (run1frac + run2frac + run3frac)
    #cosrej = ((run1frac*0.58) + (run2frac*0.60) + (run3frac*0.62)) / (run1frac + run2frac + run3frac)
    
    nueff_mu = 1.0
    nueff_nomu = 1.0
    cosrej = 0.0
    
    w = []
    for i in range(len(is_ext)):
        if is_ext[i]:
            w.append(POT_factor[i] * (1.0 - cosrej)) #0.5*
        elif is_mccosmic[i]:
            w.append(weight_cv[i]*weight_spline[i]*POT_factor[i] * (1.0 - cosrej)) #0.5*
        elif is_data[i]:
            w.append(POT_factor[i])
        elif has_muon[i]:
            w.append(weight_cv[i]*weight_spline[i] * POT_factor[i] * nueff_mu) #0.83 * 
        else:
            w.append(weight_cv[i]*weight_spline[i] * POT_factor[i] * nueff_nomu) #0.83 *     
        if w[i] <= 0. or w[i] > 30. or np.isnan(w[i]): # something went wrong with the saved weights.:
            w[i] =  POT_factor[i] * nueff_mu #0.83 * 
            
    all_df["weights"] = w

    return w


###
def PassSelection(selection, numu_score, other_score, ncpi0_score, nue_score, num_shw, num_pro, r, s, e):
    #returns a boolean array that indicates if events pass selection
    p = False
    if selection=="numu_sideband" and numu_score < 0.1 and numu_score > -20.0:
        p = True
    elif selection=="other_sideband" and numu_score > 0.1 and other_score < -0.4 and other_score > -20.0:
        p = True
    elif selection=="ncpi0_sideband" and numu_score > 0.1 and other_score > -0.4 and ncpi0_score < -0.4 and ncpi0_score > -20.0:
        p = True
    elif selection=="nue_sideband" and numu_score > 0.1 and other_score > -0.4 and ncpi0_score > -0.4 and nue_score < -3.0 and nue_score > -20.0 and num_shw==1:
        p = True
    elif selection=="eff" and numu_score > 0.1 and other_score > -0.4 and ncpi0_score > -0.4 and nue_score > -3.0 and num_shw==1:
        p = True
    elif selection=="eff_numu" and numu_score > 0.1:
        p = True
    elif selection=="eff_other" and numu_score > 0.1 and other_score > -0.4:
        p = True
    elif selection=="eff_ncpi0" and numu_score > 0.1 and other_score > -0.4 and ncpi0_score > -0.4 :
        p = True
    elif selection=="eff_nue" and numu_score > 0.1 and other_score > -0.4 and ncpi0_score > -0.4 and nue_score > -3.0:
        p = True
    elif selection=="pur" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1:
        p = True
    elif selection=="singshw" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and num_shw==1:
        p = True
    elif selection=="pur_numu" and numu_score > 0.4:
        p = True
    elif selection=="pur_other" and numu_score > 0.4 and other_score > 0.2:
        p = True
    elif selection=="pur_ncpi0" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05:
        p = True 
    elif selection=="pur_nue" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1:
        p = True
    elif selection=="0p" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro==0:
        p = True
    elif selection=="Np" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro>0:
        p = True
    elif selection=="noother" and numu_score > 0.4 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1:
        p = True
    elif selection=="noother_0p" and numu_score > 0.4 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro==0:
        p = True
    elif selection=="noother_Np" and numu_score > 0.4 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro>0:
        p = True
    elif selection=="lessother_0p" and numu_score > 0.4 and other_score > 0.0 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro==0:
        p = True
    elif selection=="moreother_0p" and numu_score > 0.4 and other_score > 1.0 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1 and num_pro==0:
        p = True
    elif selection=="allshw" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0:
        p = True
    elif selection=="all":
        p = True
    elif selection=="preselection" and numu_score>-15.0:
        p = True
    elif selection=="test" and numu_score > 0.4 and other_score > 0.2 and ncpi0_score > 0.5 and nue_score > -1.0 and num_shw==1:
        p = True
    elif selection=="pawel":
        # Open the file
        with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_pawel.txt', 'r') as file1:
            # Create a dictionary to hold the events
            #events = {}
            # Read the file line by line
            for line in file1:
                r_p, s_p, e_p, energy_p = map(float, line.split()) # Split the line into parts
                # Store in the dictionary
                #events[(r, s, e)] = energy
                if (r_p==r and s_p==s and e_p==e):
                    p = True
                    break
    elif selection=="pawel_overlap":
        # Open the file
        if (numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1):
            with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_pawel.txt', 'r') as file1:
                # Create a dictionary to hold the events
                #events = {}
                # Read the file line by line
                for line in file1:
                    r_p, s_p, e_p, energy_p = map(float, line.split()) # Split the line into parts
                    # Store in the dictionary
                    #events[(r, s, e)] = energy
                    if (r_p==r and s_p==s and e_p==e):
                        p = True
                        break
    elif selection=="pawel_nonoverlap":
        # Open the file
        if ( not (numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1)):
            with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_pawel.txt', 'r') as file1:
                # Create a dictionary to hold the events
                #events = {}
                # Read the file line by line
                for line in file1:
                    r_p, s_p, e_p, energy_p = map(float, line.split()) # Split the line into parts
                    # Store in the dictionary
                    #events[(r, s, e)] = energy
                    if (r_p==r and s_p==s and e_p==e):
                        p = True
                        break
    elif selection=="pawel_nonoverlap_1shw":
        # Open the file
        if ( not(numu_score > 0.4 and other_score > 0.2 and ncpi0_score > -0.05 and nue_score > -1.0 and num_shw==1) and num_shw==1):
            with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_pawel.txt', 'r') as file1:
                # Create a dictionary to hold the events
                #events = {}
                # Read the file line by line
                for line in file1:
                    r_p, s_p, e_p, energy_p = map(float, line.split()) # Split the line into parts
                    # Store in the dictionary
                    #events[(r, s, e)] = energy
                    if (r_p==r and s_p==s and e_p==e):
                        p = True
                        break
    elif selection=="mark":
        # Open the file
        with open('/home/erin/Documents/MicroBoone/consistency_checks/glee_epem_rse.txt', 'r') as file2:
            # Create a dictionary to hold the events
            #events = {}
            # Read the file line by line
            for line in file2:
                r_p, s_p, e_p = map(float, line.split()) # Split the line into parts
                # Store in the dictionary
                #events[(r, s, e)] = energy
                if (r_p==r and s_p==s and e_p==e):
                    p = True
                    break
    elif selection=="mark_expanded":
        # Open the file
        with open('/home/erin/Documents/MicroBoone/consistency_checks/glee_epem_expanded_angle_rse.txt', 'r') as file3:
            # Create a dictionary to hold the events
            #events = {}
            # Read the file line by line
            for line in file3:
                r_p, s_p, e_p = map(float, line.split()) # Split the line into parts
                # Store in the dictionary
                #events[(r, s, e)] = energy
                if (r_p==r and s_p==s and e_p==e):
                    p = True
                    break
    elif selection=="infv_bdt_pass":
        # Open the file
        if  numu_score > -15.0:
            with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_infv_bdt_score.txt', 'r') as file4:
                # Create a dictionary to hold the events
                #events = {}
                # Read the file line by line
                for line in file4:
                    r_p, s_p, e_p, score_p = map(float, line.split()) # Split the line into parts
                    # Store in the dictionary
                    #events[(r, s, e)] = energy
                    if (score_p>=1.0 and e_p==e and s_p==s and r_p==r):
                        p = True
                        break
    elif selection=="infv_bdt_notpass":
        # Open the file
        if  numu_score > -15.0:
            with open('/home/erin/Documents/MicroBoone/consistency_checks/rse_infv_bdt_score.txt', 'r') as file4:
                # Create a dictionary to hold the events
                #events = {}
                # Read the file line by line
                for line in file4:
                    r_p, s_p, e_p, score_p = map(float, line.split()) # Split the line into parts
                    # Store in the dictionary
                    #events[(r, s, e)] = energy
                    if (score_p<1.0 and e_p==e and s_p==s and r_p==r):
                        p = True
                        break
            
        
    
    return p

###
def MakeDataMCPlot(var_data, var_sig, var_bkg, bin_width, start_edge, end_edge, title, x_label, y_label, 
                  plotlog, changey, y_lim, selection):
    
    #function to make a data mc plot for a variable, will use whatever cut value and part of chain comes before
        #the call to the function
    #inputs:
        #var_data/sig/bkg: array of variable values for data/sig/bkg, array
        #bin_width: desired bin width (in variable units), float
        #start_edge: x-axis start value (inclusive), float or int
        #end_edge: x-axis end value (exclusive), float or int
        #title: title of plot, string
        #x_label: x-axis label, string
        #y_label: y-axis label, string
        #plotlog: if true plot y as log, bool
        #changey: if true change y, bool
        #y_lim: if exaand y true, what to set y to
        
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1
    
    h_data = ROOT.gROOT.FindObject("h_data")
    h_ext = ROOT.gROOT.FindObject("h_ext")
    h_dirt = ROOT.gROOT.FindObject("h_dirt")
    h_cos = ROOT.gROOT.FindObject("h_cos")
    h_outFV = ROOT.gROOT.FindObject("h_outFV")
    h_numuCCpi0 = ROOT.gROOT.FindObject("h_numuCCpi0")
    h_numuCC = ROOT.gROOT.FindObject("h_numuCC")
    h_NCpi0 = ROOT.gROOT.FindObject("h_NCpi0")
    h_NC = ROOT.gROOT.FindObject("h_NC")
    h_nueCC = ROOT.gROOT.FindObject("h_nueCC")
    h_NCpi1g = ROOT.gROOT.FindObject("h_NCpi1g")
    h_NCdel = ROOT.gROOT.FindObject("h_NCdel")
    h_NCother = ROOT.gROOT.FindObject("h_NCother")
    h_numuCC1g = ROOT.gROOT.FindObject("h_numuCC1g")
    h_out1g = ROOT.gROOT.FindObject("h_out1g")
    if h_data != None:
        h_data.Delete()
    if h_numuCC1g != None:
        h_ext.Delete()
        h_dirt.Delete()
        h_cos.Delete()
        h_outFV.Delete()
        h_numuCCpi0.Delete()
        h_numuCC.Delete()
        h_NCpi0.Delete()
        h_NC.Delete()
        h_nueCC.Delete()
        h_NCpi1g.Delete()
        h_NCdel.Delete()
        h_NCother.Delete()
        h_numuCC1g.Delete()
        h_out1g.Delete()
    
    h_data = ROOT.TH1F('h_data', title, bin_num, start_edge, end_edge)
    h_ext = ROOT.TH1F('h_ext', title, bin_num, start_edge, end_edge)
    h_dirt = ROOT.TH1F('h_dirt', title, bin_num, start_edge, end_edge)
    h_cos = ROOT.TH1F('h_cos', title, bin_num, start_edge, end_edge)
    h_outFV = ROOT.TH1F('h_outFV', title, bin_num, start_edge, end_edge)
    h_numuCCpi0 = ROOT.TH1F('h_numuCCpi0', title, bin_num, start_edge, end_edge)
    h_numuCC = ROOT.TH1F('h_numuCC', title, bin_num, start_edge, end_edge)
    h_NCpi0 = ROOT.TH1F('h_NCpi0', title, bin_num, start_edge, end_edge)
    h_NC = ROOT.TH1F('h_NC', title, bin_num, start_edge, end_edge)
    h_nueCC = ROOT.TH1F('h_nueCC', title, bin_num, start_edge, end_edge)
    h_NCpi1g = ROOT.TH1F('h_NCpi1g', title, bin_num, start_edge, end_edge)
    h_NCdel = ROOT.TH1F('h_NCdel', title, bin_num, start_edge, end_edge)
    h_NCother = ROOT.TH1F('h_NCother', title, bin_num, start_edge, end_edge)
    h_numuCC1g = ROOT.TH1F('h_numuCC1g', title, bin_num, start_edge, end_edge)
    h_out1g = ROOT.TH1F('h_out1g', title, bin_num, start_edge, end_edge)
    
    selected_var_sig = []
    selected_var_bkg = []
    selected_var_data = []
    
    selected_w_sig = []
    selected_w_bkg = []
    selected_w_data = []
    
    selected_true_event_type_sig = []
    selected_true_event_type_bkg = []
    selected_true_event_type_data = []

    for i in range(0, len(e_sig)):
        if(PassSelection(selection, single_photon_numu_score_sig[i], single_photon_other_score_sig[i], single_photon_ncpi0_score_sig[i], single_photon_nue_score_sig[i], num_shw_sig[i], num_pro_sig[i], r_sig[i], s_sig[i], e_sig[i])):
            selected_var_sig.append(var_sig[i])
            selected_w_sig.append(weights_sig[i])
            selected_true_event_type_sig.append(true_event_type_sig[i])

    for i in range(0, len(e_bkg)):
        if(PassSelection(selection, single_photon_numu_score_bkg[i], single_photon_other_score_bkg[i], single_photon_ncpi0_score_bkg[i], single_photon_nue_score_bkg[i], num_shw_bkg[i], num_pro_bkg[i], r_bkg[i], s_bkg[i], e_bkg[i])):
            selected_var_bkg.append(var_bkg[i])
            selected_w_bkg.append(weights_bkg[i])
            selected_true_event_type_bkg.append(true_event_type_bkg[i])

    for i in range(0, len(e_data)):
        if(PassSelection(selection, single_photon_numu_score_data[i], single_photon_other_score_data[i], single_photon_ncpi0_score_data[i], single_photon_nue_score_data[i], num_shw_data[i], num_pro_data[i], r_data[i], s_data[i], e_data[i])):
            selected_var_data.append(var_data[i])
            selected_w_data.append(weights_data[i])
            selected_true_event_type_data.append(true_event_type_data[i])

    # picking out specific backgrounds from the selected events

    selected_data_var = selected_var_data
    selected_data_w = selected_w_data

    selected_ext_var = []
    selected_dirt_var = []
    selected_cos_var = []
    selected_outFV_var = []
    selected_numuCCpi0_var = []
    selected_numuCC_var = []
    selected_NCpi0_var = []
    selected_NC_var = []
    selected_nueCC_var = []
    selected_NCpi1g_var = []
    selected_NCdel_var = []
    selected_NCother_var = []
    selected_numuCC1g_var = []
    selected_out1g_var = []
    
    selected_ext_w = []
    selected_dirt_w = []
    selected_cos_w = []
    selected_outFV_w = []
    selected_numuCCpi0_w = []
    selected_numuCC_w = []
    selected_NCpi0_w = []
    selected_NC_w = []
    selected_nueCC_w = []
    selected_NCpi1g_w = []
    selected_NCdel_w = []
    selected_NCother_w = []
    selected_numuCC1g_w = []
    selected_out1g_w = []


    for i in range(len(selected_var_data)):
        h_data.Fill(selected_var_data[i],selected_w_data[i])
    for i in range(len(selected_var_bkg)):
        if selected_true_event_type_bkg[i]==12:
            selected_ext_var.append(selected_var_bkg[i])
            selected_ext_w.append(selected_w_bkg[i])
            h_ext.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==11:
            selected_dirt_var.append(selected_var_bkg[i])
            selected_dirt_w.append(selected_w_bkg[i])
            h_dirt.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==10:
            selected_cos_var.append(selected_var_bkg[i])
            selected_cos_w.append(selected_w_bkg[i])
            h_cos.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==9:
            selected_outFV_var.append(selected_var_bkg[i])
            selected_outFV_w.append(selected_w_bkg[i])
            h_outFV.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==8: 
            selected_numuCCpi0_var.append(selected_var_bkg[i])
            selected_numuCCpi0_w.append(selected_w_bkg[i])
            h_numuCCpi0.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==7:
            selected_numuCC_var.append(selected_var_bkg[i])
            selected_numuCC_w.append(selected_w_bkg[i])
            h_numuCC.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==6:
            selected_NCpi0_var.append(selected_var_bkg[i])
            selected_NCpi0_w.append(selected_w_bkg[i])
            h_NCpi0.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==5:
            selected_NC_var.append(selected_var_bkg[i])
            selected_NC_w.append(selected_w_bkg[i])
            h_NC.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==4: 
            selected_nueCC_var.append(selected_var_bkg[i])
            selected_nueCC_w.append(selected_w_bkg[i])
            h_nueCC.Fill(selected_var_bkg[i],selected_w_bkg[i])

        else:
            print("There is an unknown additional background type")
            #print(selected_is_CC_bkg[i])
            print(selected_true_event_type_bkg[i])
            #print(selected_nu_Pdg_bkg[i])

    for i in range(len(selected_var_sig)):
        if selected_true_event_type_sig[i]==3: 
            selected_NCpi1g_var.append(selected_var_sig[i])
            selected_NCpi1g_w.append(selected_w_sig[i])
            h_NCpi1g.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==2: 
            selected_NCdel_var.append(selected_var_sig[i])
            selected_NCdel_w.append(selected_w_sig[i])
            h_NCdel.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==1: 
            selected_NCother_var.append(selected_var_sig[i])
            selected_NCother_w.append(selected_w_sig[i])
            h_NCother.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==0: 
            selected_numuCC1g_var.append(selected_var_sig[i])
            selected_numuCC1g_w.append(selected_w_sig[i])
            h_numuCC1g.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==111: 
            selected_out1g_var.append(selected_var_sig[i])
            selected_out1g_w.append(selected_w_sig[i])
            h_out1g.Fill(selected_var_sig[i],selected_w_sig[i])
            
    
    root_hists = [h_data, h_cos, h_ext, h_dirt, h_outFV, h_NCpi0, h_numuCCpi0, h_NC,h_numuCC, h_nueCC, 
                  h_NCpi1g, h_NCdel, h_NCother, h_numuCC1g, h_out1g]    
            
    
    # make the plots
    
    counts_sig_true, bin_edges, plot = plt.hist(var_sig, weights=weights_sig[:])
    counts_bkg_true, bin_edges, plot = plt.hist(var_bkg, weights=weights_bkg[:])

    counts_sig_true_unweighted, bin_edges, plot = plt.hist(var_sig)
    counts_bkg_true_unweighted, bin_edges, plot = plt.hist(var_bkg)

    counts_sig_sel_unweighted, bin_edges, plot = plt.hist(selected_var_sig)
    counts_bkg_sel_unweighted, bin_edges, plot = plt.hist(selected_var_bkg)

    counts_sig_sel, bin_edges, plot = plt.hist(selected_var_sig, weights=selected_w_sig[:])
    counts_bkg_sel, bin_edges, plot = plt.hist(selected_var_bkg, weights=selected_w_bkg[:])
    
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1

    plt.clf()
    plt.figure(dpi=100)

    pred_var = [selected_cos_var, selected_ext_var, selected_dirt_var, selected_outFV_var, selected_NCpi0_var,
                selected_numuCCpi0_var, selected_NC_var, selected_numuCC_var, selected_nueCC_var, 
                selected_NCpi1g_var, selected_NCdel_var, selected_NCother_var, selected_numuCC1g_var, selected_out1g_var]
    
    mc_weights = [selected_cos_w, selected_ext_w, selected_dirt_w, selected_outFV_w, selected_NCpi0_w,
                selected_numuCCpi0_w, selected_NC_w, selected_numuCC_w, selected_nueCC_w, 
                selected_NCpi1g_w, selected_NCdel_w, selected_NCother_w, selected_numuCC1g_w, selected_out1g_w]
    
    mc_labels = ["MC cosmic bkg("+str((round(sum(selected_cos_w),2)))+")",
                 "beam-off bkg("+str((round(sum(selected_ext_w),2)))+")",
                "dirt bkg("+str((round(sum(selected_dirt_w),2)))+")",
                "out of FV bkg("+str((round(sum(selected_outFV_w),2)))+")",
                "NC #pi^{0} bkg("+str((round(sum(selected_NCpi0_w),2)))+")",
                "#nu_{#mu}CC #pi^{0} bkg("+str((round(sum(selected_numuCCpi0_w),2)))+")",
                "NC bkg({})".format((round(sum(selected_NC_w),2))),
                "#nu_{#mu}CC bkg("+str((round(sum(selected_numuCC_w),2)))+")",
                "#nu_{e}CC bkg("+str((round(sum(selected_nueCC_w),2)))+")",
                "NC #pi^{0} 1#gamma("+str((round(sum(selected_NCpi1g_w),2)))+")",
                "NC #Delta 1#gamma("+str((round(sum(selected_NCdel_w),2)))+")",
                "NC Other 1#gamma("+str((round(sum(selected_NCother_w),2)))+")",
                "#nu_{#mu}CC 1#gamma #mu<100MeV("+str(round(sum(selected_numuCC1g_w),2))+")",
                "out of FV 1#gamma("+str(round(sum(selected_out1g_w),2))+")"]
    

    fig_e,(ex1,ex2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[4,1]})

    mc_var_hist,mc_bins_var,patches_var = ex1.hist(pred_var, bins=bin_edges, alpha=0.7, weights=mc_weights, 
                                                      histtype='barstacked', 
                                  color=colors, range=(start_edge,end_edge), label=mc_labels)


    for patch_set, hatch in zip(patches_var, hatches):
        plt.setp(patch_set, hatch=hatch)

    hist_mc_var_sig, mc_bin_edges = np.histogram(selected_var_sig, bins=bin_edges, range=(start_edge,end_edge), 
                                                 weights=selected_w_sig)
    hist_mc_var_bkg, mc_bin_edges = np.histogram(selected_var_bkg, bins=bin_edges, range=(start_edge,end_edge), 
                                                 weights=selected_w_bkg)

    #sumw2 error for mc
    bins_mc_error = np.digitize(selected_var_sig+selected_var_bkg, bin_edges)
    selected_w_mc = selected_w_sig+selected_w_bkg
    error_mc = []
    # access elements
    for i_bin in range(len(bin_edges)-1):
        bin_ws = np.where(bins_mc_error==i_bin+1,selected_w_mc,0)
        # error of bin
        error_mc.append(np.sqrt(np.sum(bin_ws**2.)))

    data_var_hist, data_bin_edges = np.histogram(selected_data_var, bins=bin_edges, range=(start_edge,end_edge), 
                                                    weights=selected_data_w)

    bin_centers = []

    for i in range(len(bin_edges)-1):
        bin_centers.append(((bin_edges[i+1] - bin_edges[i])/2)+bin_edges[i])

    selected_data_error = np.sqrt(data_var_hist)

    data_var_plot = ex1.errorbar(bin_centers, data_var_hist, yerr=selected_data_error, 
                                    color='black',
                            label='BNB Data ({})'.format(int(sum(selected_data_w))),
                            fmt='o', markersize=2, capsize=1.5, elinewidth=1, capthick=1)


    # Plot a dashed line at 1 on the ratio plot
    ex2.plot(bin_edges, [1. for x in bin_edges], "k--", linewidth=0.75)

    #bottom ratio plot
    hist_mc_var = hist_mc_var_sig+hist_mc_var_bkg
    var_data_ratio = np.divide(data_var_hist,hist_mc_var)
    selected_data_ratio_error = np.divide(selected_data_error,data_var_hist)
    for i in range(len(var_data_ratio)):
        if data_var_hist[i]==0.:
            var_data_ratio[i] = 0.
            selected_data_ratio_error[i] = 3.
        if hist_mc_var[i]==0.:
            var_data_ratio[i] = data_var_hist[i]
            selected_data_ratio_error[i] = selected_data_error[i]
        if data_var_hist[i]==0 and hist_mc_var[i]==0:
            var_data_ratio[i] = 1.
            selected_data_ratio_error[i] = 0.

    #mc error
    ex2.errorbar(bin_centers, [1. for x in bin_centers], yerr=np.divide(error_mc,hist_mc_var), 
                 color=ROOT.gROOT.GetColor(kPink).AsHexString(), label="MC Error", fmt='none', 
                 elinewidth=840/num_bins)
    #ratio + data error
    ex2.errorbar(bin_centers, var_data_ratio, yerr=selected_data_ratio_error, color='black',
                 fmt='o', markersize=2, capsize=1.5, elinewidth=1, capthick=1, label="Data/MC with Data Error")

    selected_efficiency = np.sum(counts_sig_sel) / np.sum(counts_sig_true)
    selected_purity = np.sum(counts_sig_sel) / (np.sum(counts_sig_sel) + np.sum(counts_bkg_sel))

    sum_data_mc_ratio = round(np.sum(selected_w_data)/np.sum(selected_w_sig+selected_w_bkg),2)


    ex1.set_title(title)
    if showeffpur:
        ex1.set_title(title+"\nCut value: " + str(cut_value) + ", purity: "+str(round(selected_purity, 3)) 
                      + ", sig eff: " + str(round(selected_efficiency, 3)))               
    ex1.title.set_size(title_size)
    handles, labels = ex1.get_legend_handles_labels()
    ex1.legend(handles, labels, prop={'size': 12},ncol=2,
               title = 
               '{} POT                      Stat. Uncert. Only\n$\Sigma$Data/$\Sigma$(MC+EXT)={}'.format(run1dataPOT+run2dataPOT+run3dataPOT,
                                                                                                         sum_data_mc_ratio))
    ex2.set_xlabel(x_label)
    ex1.set_ylabel(y_label)
    if plotlog:
        ex1.set_yscale("log")
    if changey:
        ex1.set_ylim(0.,y_lim)
    #ex2.legend()
    
    
    c = ROOT.TCanvas(title,title,2200,1200)
    
    h_stack = ROOT.THStack("h_stack",title)
    
    h_cos.SetLineColor(kRed+2)
    h_cos.SetFillColorAlpha(kRed+2, 0.5)
    h_cos.SetFillStyle(3004)
    h_cos.SetLineWidth(1)
    h_stack.Add(h_cos)
    
    h_ext.SetLineColor(kOrange+3)
    h_ext.SetFillColorAlpha(kOrange+3, 0.5)
    h_ext.SetFillStyle(3004)
    h_ext.SetLineWidth(1)
    h_stack.Add(h_ext)
    
    h_dirt.SetLineColor(kGray+2)
    h_dirt.SetFillColorAlpha(kGray, 0.5)
    h_dirt.SetFillStyle(3224)
    h_dirt.SetLineWidth(1)
    h_stack.Add(h_dirt)
    
    h_outFV.SetLineColor(kOrange+1)
    h_outFV.SetFillColorAlpha(kOrange+1, 0.5)
    h_outFV.SetFillStyle(3224)
    h_outFV.SetLineWidth(1)
    h_stack.Add(h_outFV)
    
    h_NCpi0.SetLineColor(38)
    h_NCpi0.SetFillColorAlpha(38, 0.5)
    h_NCpi0.SetFillStyle(1001)
    h_NCpi0.SetLineWidth(1)
    h_stack.Add(h_NCpi0)

    h_numuCCpi0.SetLineColor(30)
    h_numuCCpi0.SetFillColorAlpha(30, 0.5)
    h_numuCCpi0.SetFillStyle(1001)
    h_numuCCpi0.SetLineWidth(1)
    h_stack.Add(h_numuCCpi0)
    
    h_NC.SetLineColor(kOrange+1)
    h_NC.SetFillColorAlpha(kOrange+1, 0.5)
    h_NC.SetFillStyle(1001)
    h_NC.SetLineWidth(1)
    h_stack.Add(h_NC)
    
    h_numuCC.SetLineColor(kAzure+6)
    h_numuCC.SetFillColorAlpha(kAzure+6, 0.5)
    h_numuCC.SetFillStyle(1001)
    h_numuCC.SetLineWidth(1)
    h_stack.Add(h_numuCC)

    h_nueCC.SetLineColor(kGreen+1)
    h_nueCC.SetFillColorAlpha(kGreen+1, 0.5)
    h_nueCC.SetFillStyle(1001)
    h_nueCC.SetLineWidth(1)
    h_stack.Add(h_nueCC)
        
    h_NCpi1g.SetLineColor(kPink+5)
    h_NCpi1g.SetFillColorAlpha(kPink+5, 0.5)
    h_NCpi1g.SetFillStyle(1001)
    h_NCpi1g.SetLineWidth(1)
    h_stack.Add(h_NCpi1g)
        
    h_NCdel.SetLineColor(kPink-6)
    h_NCdel.SetFillColorAlpha(kPink-6, 0.5)
    h_NCdel.SetFillStyle(1001)
    h_NCdel.SetLineWidth(1)
    h_stack.Add(h_NCdel)

    h_NCother.SetLineColor(kPink-8)
    h_NCother.SetFillColorAlpha(kPink-8, 0.5)
    h_NCother.SetFillStyle(1001)
    h_NCother.SetLineWidth(1)
    h_stack.Add(h_NCother)

    h_numuCC1g.SetLineColor(kPink-7)
    h_numuCC1g.SetFillColorAlpha(kPink-7, 0.5)
    h_numuCC1g.SetFillStyle(1001)
    h_numuCC1g.SetLineWidth(1)
    h_stack.Add(h_numuCC1g)

    h_out1g.SetLineColor(kPink)
    h_out1g.SetFillColorAlpha(kPink, 0.5)
    h_out1g.SetFillStyle(1001)
    h_out1g.SetLineWidth(1)
    h_stack.Add(h_out1g)

    h_stack.Draw()
    
    h_data.SetFillColor(kWhite)
    h_data.SetLineColor(kBlack)
    h_data.SetLineWidth(3)
    if h_data.GetMaximum() <= h_stack.GetMaximum():
        h_data.SetMaximum(h_stack.GetMaximum()+10.)
    #h_data.Draw("hist")
    
    stackHists = h_stack.GetHists()
    tmpHist = stackHists[0].Clone()
    tmpHist.Reset()
    for hist in stackHists:
          tmpHist.Add(hist)
    tmpHist.SetLineColor(kWhite)
    tmpHist.SetFillColor(kWhite)
    tmpHist.SetMarkerColor(kWhite)
    tmpHist.SetLineWidth(0)
    rp = ROOT.TRatioPlot(h_data, tmpHist)
    #c.SetTicks(0, 1)
    rp.Draw("AXIS")
    h_data.SetTitle("")
    h_data.GetXaxis().SetTitle(x_label)
    h_stack.GetXaxis().SetTitle(x_label)
    h_data.GetXaxis().SetLabelSize(label_size)
    h_stack.GetXaxis().SetLabelSize(label_size)
    c.Modified() 
    c.Update()
    rp.GetLowerRefGraph().GetYaxis().SetTitle("Data/(MC+EXT)")
    rp.GetLowerRefGraph().GetYaxis().SetLabelSize(label_size)
    rp.GetLowerRefGraph().GetYaxis().SetRangeUser(0.,2.5)
    #rp.GetLowerRefGraph().GetYaxis().SetNdivisions(5)
    tmpHist.SetFillColor(kPink)
    tmpHist.SetFillStyle(3005)
    tmpHist.Divide(tmpHist,tmpHist,1.0,1.0)
    bottom = rp.GetLowerPad()
    bottom.cd()
    tmpHist.Draw("Same E2")
    tmpHist.GetYaxis().SetTitle("Data/(MC+EXT)")
    tmpHist.GetYaxis().SetLabelSize(label_size)
    botleg = ROOT.TLegend(0.75,0.75,0.9,0.9)
    #botleg.SetNColumns(2)
    botleg.SetTextSize(0.07)
    botleg.AddEntry(tmpHist,"MC+EXT Uncertainty")
    botleg.Draw()
    
    p = rp.GetUpperPad()
    p.cd()
    
    #pvalue = h_data.Chi2Test(tmpHist)
    #print(pvalue)
    #chi2 = round(h_data.Chi2Test(tmpHist,"CHI2/NDF"),2)
    
    leg = ROOT.TLegend(legx1,legy1,legx2,legy2)
    leg.SetNColumns(2)
    leg.SetHeader("MicroBooNE Preliminary","C")
    leg.AddEntry(0,"{} POT  Stat. Uncert. Only".format(run1dataPOT+run2dataPOT+run3dataPOT),"")
    leg.AddEntry(0,"#SigmaData/#Sigma(MC+EXT)={}".format(sum_data_mc_ratio),"")
    #leg.AddEntry(0, "#chi^2/dof = {}".format(chi2),"")
    mc_labels.insert(0,'BNB Data ({})'.format(int(sum(selected_data_w))))
    for h,l in zip(root_hists, mc_labels):
        leg.AddEntry(h,l)  
    header = leg.GetListOfPrimitives().First()
    header.SetTextAlign(22)
    #header.SetTextColor(2)
    header.SetTextSize(.05)
    
    h_data.Draw("AE")
    h_stack.Draw("hist same")
    h_data.Draw("AE same")
    leg.Draw()
    if plotlog:
        p.SetLogy()
    p.Modified() 
    p.Update()
    if var_data == single_photon_other_score_data:
        cut=ROOT.TLine(0.2,p.GetUymin(),0.2,p.GetUymax())
        cut.SetLineColor(kBlack)
        cut.SetLineStyle(9)
        cut.SetLineWidth(3)
        cut.Draw("same")
    c.Update()
    
    c.Update()
    #c.Draw()
    c.Print('plots/root_plots/'+plot_folder+'/datamc/'+title+'.png')
    
    print(title)
    
    return fig_e


###
def MakeDataPlot(var_data, bin_width, start_edge, end_edge, title, x_label, y_label, 
                  plotlog, changey, y_lim, selection):
    
    #function to make a data plot for a variable, will use cuts given by "selection"
    #inputs:
        #var_data: array of variable values for data, array
        #bin_width: desired bin width (in variable units), float
        #start_edge: x-axis start value (inclusive), float or int
        #end_edge: x-axis end value (exclusive), float or int
        #title: title of plot, string
        #x_label: x-axis label, string
        #y_label: y-axis label, string
        #plotlog: if true plot y as log, bool
        #changey: if true change y, bool
        #y_lim: if exaand y true, what to set y to
        #selection: which selection cuts to use (example: 0p, ncpi0_sideband, etc.)
        
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1
    
    h_data = ROOT.gROOT.FindObject("h_data")
    if h_data != None:
        h_data.Delete()
    
    h_data = ROOT.TH1F('h_data', title, bin_num, start_edge, end_edge)
    
    selected_var_data = []
    
    selected_w_data = []
    
    selected_true_event_type_data = []

    for i in range(0, len(e_data)):
        if(PassSelection(selection, single_photon_numu_score_data[i], single_photon_other_score_data[i], single_photon_ncpi0_score_data[i], single_photon_nue_score_data[i], num_shw_data[i], num_pro_data[i], r_data[i], s_data[i], e_data[i])):
            selected_var_data.append(var_data[i])
            selected_w_data.append(weights_data[i])
            selected_true_event_type_data.append(true_event_type_data[i])

    # picking out specific backgrounds from the selected events

    selected_data_var = selected_var_data
    selected_data_w = selected_w_data


    for i in range(len(selected_var_data)):
        h_data.Fill(selected_var_data[i],selected_w_data[i])
            
    
    #root_hists = [h_out1g, h_numuCC1g,h_NCother,h_NCdel,h_NCpi1g,h_nueCC,h_NC,h_NCpi0,h_numuCC,h_numuCCpi0,
    #         h_outFV,h_cos,h_dirt,h_ext, h_data]
            
    
    # make the plots
    
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1

    plt.clf()
    plt.figure(dpi=100)
    
    fig_e,(ex1,ex2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[4,1]})
    

    data_var_hist, data_bin_edges = np.histogram(selected_data_var, bins=bin_edges, range=(start_edge,end_edge), 
                                                    weights=selected_data_w)

    bin_centers = []

    for i in range(len(bin_edges)-1):
        bin_centers.append(((bin_edges[i+1] - bin_edges[i])/2)+bin_edges[i])

    selected_data_error = np.sqrt(data_var_hist)

    data_var_plot = ex1.errorbar(bin_centers, data_var_hist, yerr=selected_data_error, 
                                    color='black',
                            label='BNB Data ({})'.format(int(sum(selected_data_w))),
                            fmt='o', markersize=2, capsize=1.5, elinewidth=1, capthick=1)
            
    ex1.title.set_size(title_size)
    handles, labels = ex1.get_legend_handles_labels()
    ex1.legend(reversed(handles), reversed(labels), prop={'size': 12},ncol=2,
               title = 
               '{} POT'.format(run1dataPOT+run2dataPOT+run3dataPOT))
    
    ex2.set_xlabel(x_label)
    ex1.set_ylabel(y_label)
    if plotlog:
        ex1.set_yscale("log")
    if changey:
        ex1.set_ylim(0.,y_lim)
    #ex2.legend()

    if var_data == time_data:
        #ROOT.gStyle.SetOptFit(1011)
        # Create the fit function
        fitFunc = ROOT.TF1("fitFunc", "[0] * exp(-0.5 * ((x - [1]) / [2])**2) + [3]", -9, 9)

        # Set initial parameter values
        fitFunc.SetParameters(100, 0, 1, 10)

        # Set parameter names
        fitFunc.SetParNames("Amplitude", "Mean", "Sigma", "Offset")

        # Fit the histogram
        h_data.Fit("fitFunc", "M")

        
    
    
    c = ROOT.TCanvas(title,title,2200,1200)
    
    h_data.SetFillColor(kWhite)
    h_data.SetLineColor(kBlack)
    h_data.SetLineWidth(3)
    h_data.Draw()
    h_data.SetTitle(title)
    h_data.GetXaxis().SetTitle(x_label)
    h_data.GetYaxis().SetTitle(y_label)
    h_data.GetXaxis().SetLabelSize(label_size)
    h_data.GetYaxis().SetLabelSize(label_size)
    c.Modified() 
    c.Update()
    
    leg = ROOT.TLegend(legx1,legy1,legx2,legy2)
    #leg.SetNColumns(2)
    leg.SetHeader("MicroBooNE Preliminary","C")
    leg.AddEntry(0,"{} POT".format(run1dataPOT+run2dataPOT+run3dataPOT),"")
    #leg.AddEntry(0,"#SigmaData/#Sigma(MC+EXT)={}".format(sum_data_mc_ratio),"")
    #leg.AddEntry(0, "#chi^2/dof = {}".format(chi2),"")
    leg.AddEntry(h_data,'BNB Data ({})'.format(int(sum(selected_data_w))),"lp")  
    header = leg.GetListOfPrimitives().First()
    header.SetTextAlign(22)
    #header.SetTextColor(2)
    header.SetTextSize(.05)
    
    h_data.Draw("E")
    #h_data.Draw("AE same")
    if var_data != time_data:
        leg.Draw()
    if plotlog:
        p.SetLogy()
    c.Modified() 
    c.Update()
    #c.Draw()
    c.Print('plots/root_plots/'+plot_folder+'/data/'+title+'.png')
    
    print(title)
    
    return h_data

###
def MakeMCPlot(var_sig, var_bkg, bin_width, start_edge, end_edge, title, x_label, y_label, selection, systdir=""):
    
    #function to make a mc plot for a variable, will use whatever cut value and part of chain comes before
        #the call to the function
    #inputs:
        #var_sig/bkg: array of variable values for sig/bkg, array
        #bin_width: desired bin width (in variable units), float
        #start_edge: x-axis start value (inclusive), float or int
        #end_edge: x-axis end value (exclusive), float or int
        #title: title of plot, string
        #x_label: x-axis label, string
        #y_label: y-axis label, string
        
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1
        
    h_ext = ROOT.gROOT.FindObject("h_ext")
    h_dirt = ROOT.gROOT.FindObject("h_dirt")
    h_cos = ROOT.gROOT.FindObject("h_cos")
    h_outFV = ROOT.gROOT.FindObject("h_outFV")
    h_numuCCpi0 = ROOT.gROOT.FindObject("h_numuCCpi0")
    h_numuCC = ROOT.gROOT.FindObject("h_numuCC")
    h_NCpi0 = ROOT.gROOT.FindObject("h_NCpi0")
    h_NC = ROOT.gROOT.FindObject("h_NC")
    h_nuepi0 = ROOT.gROOT.FindObject("h_nueCC")
    h_NCpi1g = ROOT.gROOT.FindObject("h_NCpi1g")
    h_NCdel = ROOT.gROOT.FindObject("h_NCdel")
    h_NCother = ROOT.gROOT.FindObject("h_NCother")
    h_numuCC1g = ROOT.gROOT.FindObject("h_numuCC1g")
    h_out1g = ROOT.gROOT.FindObject("h_out1g")
    h_sig = ROOT.gROOT.FindObject("h_sig")
    h_bkg = ROOT.gROOT.FindObject("h_bkg")
    if h_ext != None:
        h_ext.Delete()
        h_dirt.Delete()
        h_cos.Delete()
        h_outFV.Delete()
        h_numuCCpi0.Delete()
        h_numuCC.Delete()
        h_NCpi0.Delete()
        h_NC.Delete()
        h_nuepi0.Delete()
        h_NCpi1g.Delete()
        h_NCdel.Delete()
        h_NCother.Delete()
        h_numuCC1g.Delete()
        h_out1g.Delete()
    if h_sig != None:
        h_sig.Delete()
        h_bkg.Delete()
    
    #h_data = ROOT.TH1F('h_data', title, bin_num, start_edge, end_edge)
    h_ext = ROOT.TH1F('h_ext', title, bin_num, start_edge, end_edge)
    h_dirt = ROOT.TH1F('h_dirt', title, bin_num, start_edge, end_edge)
    h_cos = ROOT.TH1F('h_cos', title, bin_num, start_edge, end_edge)
    h_outFV = ROOT.TH1F('h_outFV', title, bin_num, start_edge, end_edge)
    h_numuCCpi0 = ROOT.TH1F('h_numuCCpi0', title, bin_num, start_edge, end_edge)
    h_numuCC = ROOT.TH1F('h_numuCC', title, bin_num, start_edge, end_edge)
    h_NCpi0 = ROOT.TH1F('h_NCpi0', title, bin_num, start_edge, end_edge)
    h_NC = ROOT.TH1F('h_NC', title, bin_num, start_edge, end_edge)
    h_nueCC = ROOT.TH1F('h_nueCC', title, bin_num, start_edge, end_edge)
    h_NCpi1g = ROOT.TH1F('h_NCpi1g', title, bin_num, start_edge, end_edge)
    h_NCdel = ROOT.TH1F('h_NCdel', title, bin_num, start_edge, end_edge)
    h_NCother = ROOT.TH1F('h_NCother', title, bin_num, start_edge, end_edge)
    h_numuCC1g = ROOT.TH1F('h_numuCC1g', title, bin_num, start_edge, end_edge)
    h_out1g = ROOT.TH1F('h_out1g', title, bin_num, start_edge, end_edge)
    
    h_sig = ROOT.TH1F('h_sig', title, bin_num, start_edge, end_edge)
    h_bkg = ROOT.TH1F('h_bkg', title, bin_num, start_edge, end_edge)
    
    selected_var_sig = []
    selected_var_bkg = []
    
    selected_w_sig = []
    selected_w_bkg = []
    
    selected_true_event_type_sig = []
    selected_true_event_type_bkg = []

    for i in range(0, len(e_sig)):
        if(PassSelection(selection, single_photon_numu_score_sig[i], single_photon_other_score_sig[i], single_photon_ncpi0_score_sig[i], single_photon_nue_score_sig[i], num_shw_sig[i], num_pro_sig[i], r_sig[i], s_sig[i], e_sig[i])):
            selected_var_sig.append(var_sig[i])
            selected_w_sig.append(weights_sig[i])
            selected_true_event_type_sig.append(true_event_type_sig[i])

    for i in range(0, len(e_bkg)):
        if(PassSelection(selection, single_photon_numu_score_bkg[i], single_photon_other_score_bkg[i], single_photon_ncpi0_score_bkg[i], single_photon_nue_score_bkg[i], num_shw_bkg[i], num_pro_bkg[i], r_bkg[i], s_bkg[i], e_bkg[i])):
            selected_var_bkg.append(var_bkg[i])
            selected_w_bkg.append(weights_bkg[i])
            selected_true_event_type_bkg.append(true_event_type_bkg[i])

    # picking out specific backgrounds from the selected events

    selected_ext_var = []
    selected_dirt_var = []
    selected_cos_var = []
    selected_outFV_var = []
    selected_numuCCpi0_var = []
    selected_numuCC_var = []
    selected_NCpi0_var = []
    selected_NC_var = []
    selected_nueCC_var = []
    selected_NCpi1g_var = []
    selected_NCdel_var = []
    selected_NCother_var = []
    selected_numuCC1g_var = []
    selected_out1g_var = []
    
    selected_ext_w = []
    selected_dirt_w = []
    selected_cos_w = []
    selected_outFV_w = []
    selected_numuCCpi0_w = []
    selected_numuCC_w = []
    selected_NCpi0_w = []
    selected_NC_w = []
    selected_nueCC_w = []
    selected_NCpi1g_w = []
    selected_NCdel_w = []
    selected_NCother_w = []
    selected_numuCC1g_w = []
    selected_out1g_w = []


    for i in range(len(selected_var_bkg)):
        h_bkg.Fill(selected_var_bkg[i],selected_w_bkg[i])
        if selected_true_event_type_bkg[i]==12:
            selected_ext_var.append(selected_var_bkg[i])
            selected_ext_w.append(selected_w_bkg[i])
            h_ext.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==11:
            selected_dirt_var.append(selected_var_bkg[i])
            selected_dirt_w.append(selected_w_bkg[i])
            h_dirt.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==10:
            selected_cos_var.append(selected_var_bkg[i])
            selected_cos_w.append(selected_w_bkg[i])
            h_cos.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==9:
            selected_outFV_var.append(selected_var_bkg[i])
            selected_outFV_w.append(selected_w_bkg[i])
            h_outFV.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==8: 
            selected_numuCCpi0_var.append(selected_var_bkg[i])
            selected_numuCCpi0_w.append(selected_w_bkg[i])
            h_numuCCpi0.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==7:
            selected_numuCC_var.append(selected_var_bkg[i])
            selected_numuCC_w.append(selected_w_bkg[i])
            h_numuCC.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==6:
            selected_NCpi0_var.append(selected_var_bkg[i])
            selected_NCpi0_w.append(selected_w_bkg[i])
            h_NCpi0.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==5:
            selected_NC_var.append(selected_var_bkg[i])
            selected_NC_w.append(selected_w_bkg[i])
            h_NC.Fill(selected_var_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==4: 
            selected_nueCC_var.append(selected_var_bkg[i])
            selected_nueCC_w.append(selected_w_bkg[i])
            h_nueCC.Fill(selected_var_bkg[i],selected_w_bkg[i])

        else:
            print("There is an unknown additional background type")
            #print(selected_is_CC_bkg[i])
            print(selected_true_event_type_bkg[i])
            #print(selected_nu_Pdg_bkg[i])

    for i in range(len(selected_var_sig)):
        h_sig.Fill(selected_var_sig[i],selected_w_sig[i])
        if selected_true_event_type_sig[i]==3: 
            selected_NCpi1g_var.append(selected_var_sig[i])
            selected_NCpi1g_w.append(selected_w_sig[i])
            h_NCpi1g.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==2: 
            selected_NCdel_var.append(selected_var_sig[i])
            selected_NCdel_w.append(selected_w_sig[i])
            h_NCdel.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==1: 
            selected_NCother_var.append(selected_var_sig[i])
            selected_NCother_w.append(selected_w_sig[i])
            h_NCother.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==0: 
            selected_numuCC1g_var.append(selected_var_sig[i])
            selected_numuCC1g_w.append(selected_w_sig[i])
            h_numuCC1g.Fill(selected_var_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==111: 
            selected_out1g_var.append(selected_var_sig[i])
            selected_out1g_w.append(selected_w_sig[i])
            h_out1g.Fill(selected_var_sig[i],selected_w_sig[i])
            
    
    root_hists = [h_cos, h_ext, h_dirt, h_outFV, h_NCpi0, h_numuCCpi0, h_NC,h_numuCC, h_nueCC, 
                  h_NCpi1g, h_NCdel, h_NCother, h_numuCC1g, h_out1g] 
    
    # make the plots
    
    counts_sig_true, bin_edges, plot = plt.hist(var_sig, weights=weights_sig[:])
    counts_bkg_true, bin_edges, plot = plt.hist(var_bkg, weights=weights_bkg[:])

    counts_sig_true_unweighted, bin_edges, plot = plt.hist(var_sig)
    counts_bkg_true_unweighted, bin_edges, plot = plt.hist(var_bkg)

    counts_sig_sel_unweighted, bin_edges, plot = plt.hist(selected_var_sig)
    counts_bkg_sel_unweighted, bin_edges, plot = plt.hist(selected_var_bkg)

    counts_sig_sel, bin_edges, plot = plt.hist(selected_var_sig, weights=selected_w_sig[:])
    counts_bkg_sel, bin_edges, plot = plt.hist(selected_var_bkg, weights=selected_w_bkg[:])
    
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1

    plt.clf()
    plt.figure(dpi=100)

    pred_var = [selected_cos_var, selected_ext_var, selected_dirt_var, selected_outFV_var, selected_NCpi0_var,
                selected_numuCCpi0_var, selected_NC_var, selected_numuCC_var, selected_nueCC_var, 
                selected_NCpi1g_var, selected_NCdel_var, selected_NCother_var, selected_numuCC1g_var, selected_out1g_var]
    
    mc_weights = [selected_cos_w, selected_ext_w, selected_dirt_w, selected_outFV_w, selected_NCpi0_w,
                selected_numuCCpi0_w, selected_NC_w, selected_numuCC_w, selected_nueCC_w, 
                selected_NCpi1g_w, selected_NCdel_w, selected_NCother_w, selected_numuCC1g_w, selected_out1g_w]
    
    mc_labels = ["MC cosmic bkg("+str((round(sum(selected_cos_w),2)))+")",
                 "beam-off bkg("+str((round(sum(selected_ext_w),2)))+")",
                "dirt bkg("+str((round(sum(selected_dirt_w),2)))+")",
                "out of FV bkg("+str((round(sum(selected_outFV_w),2)))+")",
                "NC #pi^{0} bkg("+str((round(sum(selected_NCpi0_w),2)))+")",
                "#nu_{#mu}CC #pi^{0} bkg("+str((round(sum(selected_numuCCpi0_w),2)))+")",
                "NC bkg({})".format((round(sum(selected_NC_w),2))),
                "#nu_{#mu}CC bkg("+str((round(sum(selected_numuCC_w),2)))+")",
                "#nu_{e}CC bkg("+str((round(sum(selected_nueCC_w),2)))+")",
                "NC #pi^{0} 1#gamma("+str((round(sum(selected_NCpi1g_w),2)))+")",
                "NC #Delta 1#gamma("+str((round(sum(selected_NCdel_w),2)))+")",
                "NC Other 1#gamma("+str((round(sum(selected_NCother_w),2)))+")",
                "#nu_{#mu}CC 1#gamma #mu<100MeV("+str(round(sum(selected_numuCC1g_w),2))+")",
                "out of FV 1#gamma("+str(round(sum(selected_out1g_w),2))+")"]

    mc_var_hist,mc_bins_var,patches_var = plt.hist(pred_var, bins=bin_edges, alpha=0.7, weights=mc_weights, 
                                                      histtype='barstacked', 
                                  color=colors, range=(start_edge,end_edge), label=mc_labels)


    for patch_set, hatch in zip(patches_var, hatches):
        plt.setp(patch_set, hatch=hatch)

    hist_mc_var_sig, mc_bin_edges = np.histogram(selected_var_sig, bins=bin_edges, range=(start_edge,end_edge), 
                                                 weights=selected_w_sig)
    hist_mc_var_bkg, mc_bin_edges = np.histogram(selected_var_bkg, bins=bin_edges, range=(start_edge,end_edge), 
                                                 weights=selected_w_bkg)

    #sumw2 error for mc
    bins_mc_error = np.digitize(selected_var_sig+selected_var_bkg, bin_edges)
    selected_w_mc = selected_w_sig+selected_w_bkg
    error_mc = []
    # access elements
    for i_bin in range(len(bin_edges)-1):
        bin_ws = np.where(bins_mc_error==i_bin+1,selected_w_mc,0)
        # error of bin
        error_mc.append(np.sqrt(np.sum(bin_ws**2.)))


    bin_centers = []

    for i in range(len(bin_edges)-1):
        bin_centers.append(((bin_edges[i+1] - bin_edges[i])/2)+bin_edges[i])

   # data_var_plot = plt.errorbar(bin_centers, hist_mc_var_sig+hist_mc_var_bkg, yerr=error_mc, 
   #                                 color='red',
   #                         label='Total MC ({})'.format(int(sum(selected_w_sig+selected_w_bkg))),
   #                         fmt='o', markersize=2, capsize=1.5, elinewidth=1, capthick=1)

    
    

    plt.title(title, fontsize = title_size)
    #handles, labels = plt.axes.Axes.get_legend_handles_labels()
    plt.legend(plt.legend().legend_handles,mc_labels, prop={'size': 12},ncol=2,
               title = 
               '{} POT                      Stat. Uncert. Only'.format(run1dataPOT+run2dataPOT+run3dataPOT))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if plotlog:
        plt.yscale("log")
        
    c = ROOT.TCanvas(title,title,2200,1200)
    
    h_stack = ROOT.THStack("h_stack",title)
    
    h_cos.SetLineColor(kRed+2)
    h_cos.SetFillColorAlpha(kRed+2, 1.0)
    h_cos.SetFillStyle(3004)
    h_cos.SetLineWidth(1)
    h_stack.Add(h_cos)
    
    h_ext.SetLineColor(kOrange+3)
    h_ext.SetFillColorAlpha(kOrange+3, 1.0)
    h_ext.SetFillStyle(3004)
    h_ext.SetLineWidth(1)
    h_stack.Add(h_ext)
    
    h_dirt.SetLineColor(kGray+2)
    h_dirt.SetFillColorAlpha(kGray, 1.0)
    h_dirt.SetFillStyle(3224)
    h_dirt.SetLineWidth(1)
    h_stack.Add(h_dirt)
    
    h_outFV.SetLineColor(kOrange+1)
    h_outFV.SetFillColorAlpha(kOrange+1, 1.0)
    h_outFV.SetFillStyle(3224)
    h_outFV.SetLineWidth(1)
    h_stack.Add(h_outFV)
    
    h_NCpi0.SetLineColor(38)
    h_NCpi0.SetFillColorAlpha(38, 1.0)
    h_NCpi0.SetFillStyle(1001)
    h_NCpi0.SetLineWidth(1)
    h_stack.Add(h_NCpi0)

    h_numuCCpi0.SetLineColor(30)
    h_numuCCpi0.SetFillColorAlpha(30, 1.0)
    h_numuCCpi0.SetFillStyle(1001)
    h_numuCCpi0.SetLineWidth(1)
    h_stack.Add(h_numuCCpi0)
    
    h_NC.SetLineColor(kOrange+1)
    h_NC.SetFillColorAlpha(kOrange+1, 1.0)
    h_NC.SetFillStyle(1001)
    h_NC.SetLineWidth(1)
    h_stack.Add(h_NC)
    
    h_numuCC.SetLineColor(kAzure+6)
    h_numuCC.SetFillColorAlpha(kAzure+6, 1.0)
    h_numuCC.SetFillStyle(1001)
    h_numuCC.SetLineWidth(1)
    h_stack.Add(h_numuCC)

    h_nueCC.SetLineColor(kGreen+1)
    h_nueCC.SetFillColorAlpha(kGreen+1, 1.0)
    h_nueCC.SetFillStyle(1001)
    h_nueCC.SetLineWidth(1)
    h_stack.Add(h_nueCC)
        
    h_NCpi1g.SetLineColor(kPink+5)
    h_NCpi1g.SetFillColorAlpha(kPink+5, 1.0)
    h_NCpi1g.SetFillStyle(1001)
    h_NCpi1g.SetLineWidth(1)
    h_stack.Add(h_NCpi1g)
        
    h_NCdel.SetLineColor(kPink-6)
    h_NCdel.SetFillColorAlpha(kPink-6, 1.0)
    h_NCdel.SetFillStyle(1001)
    h_NCdel.SetLineWidth(1)
    h_stack.Add(h_NCdel)

    h_NCother.SetLineColor(kPink-8)
    h_NCother.SetFillColorAlpha(kPink-8, 1.0)
    h_NCother.SetFillStyle(1001)
    h_NCother.SetLineWidth(1)
    h_stack.Add(h_NCother)

    h_numuCC1g.SetLineColor(kPink-7)
    h_numuCC1g.SetFillColorAlpha(kPink-7, 1.0)
    h_numuCC1g.SetFillStyle(1001)
    h_numuCC1g.SetLineWidth(1)
    h_stack.Add(h_numuCC1g)

    h_out1g.SetLineColor(kPink)
    h_out1g.SetFillColorAlpha(kPink, 1.0)
    h_out1g.SetFillStyle(1001)
    h_out1g.SetLineWidth(1)
    h_stack.Add(h_out1g)

    h_stack.Draw("hist")

    #make syst errors from existing TLEE file
    if systdir != "":
        stackHists = h_stack.GetHists()
        hmc = stackHists[0].Clone()
        hmc.Reset()
        for hist in stackHists:
          hmc.Add(hist)
        hmcerror = hmc.Clone("hmcerror")
        hmcerror.Sumw2()
        #hmcerror.Scale(scalePOT)
        hmcerror.Draw("same E2")
        hmcerror.SetFillColor(kGray+2)
        hmcerror.SetFillStyle(3002)
        hmcerror.SetLineWidth(0)
        hmcerror.SetLineColor(12)
        hmcerror.SetMarkerColor(0)
        hmcerror.SetMarkerSize(0)
        #make syst errors from existing TLEE file
        GOF = {}
        sumtotalcov = 0.0
        print(f"Total uncertainty from external covariance matrix: {systdir}")
        f_cov = ROOT.TFile(systdir, "READ")

        flag_syst_flux_Xs = array('i', [0])
        flag_syst_detector = array('i', [0])
        flag_syst_additional = array('i', [0])
        flag_syst_mc_stat = array('i', [0])
        cov_lee_strength = array('d', [0])
        vc_val_GOF = ROOT.std.vector('double')()
        vc_val_GOF_NDF = ROOT.std.vector('int')()

        t_covconfig = f_cov.Get("tree")
        t_covconfig.SetBranchAddress("flag_syst_flux_Xs", flag_syst_flux_Xs)
        t_covconfig.SetBranchAddress("flag_syst_detector", flag_syst_detector)
        t_covconfig.SetBranchAddress("flag_syst_additional", flag_syst_additional)
        t_covconfig.SetBranchAddress("flag_syst_mc_stat", flag_syst_mc_stat)
        t_covconfig.SetBranchAddress("user_Lee_strength_for_output_covariance_matrix", cov_lee_strength)
        t_covconfig.SetBranchAddress("vc_val_GOF", vc_val_GOF)
        t_covconfig.SetBranchAddress("vc_val_GOF_NDF", vc_val_GOF_NDF)
        t_covconfig.GetEntry(0)

        # fill GOF map
        for vv in range(vc_val_GOF.size()):
            GOF[vv] = (vc_val_GOF[vv], vc_val_GOF_NDF[vv])

        # absolute cov matrix
        matrix_absolute_cov = f_cov.Get("matrix_absolute_cov_newworld")
        matrix_absolute_detector_cov = f_cov.Get("matrix_absolute_detector_cov_newworld")

        print(f"Cov matrix config: \n"
            f"\t syst_flux_Xs: {flag_syst_flux_Xs[0]}\n"
            f"\t syst_detector: {flag_syst_detector[0]}\n"
            f"\t syst_additional: {flag_syst_additional[0]}\n"
            f"\t syst_mc_stat: {flag_syst_mc_stat[0]}\n"
            f"\t LEE_strength: {cov_lee_strength[0]}")

        # construct a map from (obsch, bin) to cov index
        obsch_bin_index = {}
        index = 0
        #for i in range(len(map_obsch_histos)):
            # + overflow bin
            #for j in range(1, map_obsch_histos[i+1][1].GetNbinsX() + 2):
                #index += 1
                # cov index starts from 0
                #obsch_bin_index[(i+1, j)] = index - 1

        matrix_pred = f_cov.Get("matrix_pred_newworld")
        matrix_data = f_cov.Get("matrix_data_newworld")
        #for obsch, histos in map_obsch_histos.items():
        h1 = hmcerror.Clone("h1")  # error --> total uncertainty
        h2 = hmcerror.Clone("h2")  # bonus: error --> detector systematic uncertainty

        htemp_data = h1.Clone("htemp_data")
        htemp_pred = h1.Clone("htemp_pred")
        htemp_data.Reset()
        htemp_pred.Reset()
        temp_sumtotalcov = 0
        for i in range(h1.GetNbinsX() + 1):
            index = i #obsch_bin_index[(obsch, i+1)]
            total_uncertainty = matrix_absolute_cov[index][index]  # only diagonal term
            # summation of total cov
            if i != h1.GetNbinsX():  # no overflow bin in this calculation
                for j in range(h1.GetNbinsX()):
                    jndex = j #obsch_bin_index[(obsch, j+1)]
                    temp_sumtotalcov += matrix_absolute_cov[index][jndex]
            detector_uncertainty = matrix_absolute_detector_cov[index][index]  # only diagonal term
            if h1.GetBinContent(i+1) > 0:
                print(f"{i} {h1.GetBinContent(i+1)} {total_uncertainty} {ROOT.TMath.Sqrt(total_uncertainty)/h1.GetBinContent(i+1)} {h2.GetBinContent(i+1)} {detector_uncertainty}")
            h1.SetBinError(i+1, ROOT.TMath.Sqrt(total_uncertainty))
            h2.SetBinError(i+1, ROOT.TMath.Sqrt(detector_uncertainty))

            sumtotalcov = temp_sumtotalcov
            hmcerror.SetBinError(i+1, ROOT.TMath.Sqrt(total_uncertainty))
            hmcerror.Draw("E2")
            hmcerror.GetXaxis().SetTitle(x_label)
            hmcerror.GetYaxis().SetTitle(y_label)
            hmcerror.GetXaxis().SetTitleSize(label_size)
            hmcerror.GetYaxis().SetTitleSize(label_size)

    h_stack.Draw("hist same")
    if systdir != "":
        hmcerror.Draw("same E2")
    
    h_stack.GetXaxis().SetTitle(x_label)
    h_stack.GetYaxis().SetTitle(y_label)
    h_stack.GetXaxis().SetTitleSize(label_size)
    h_stack.GetYaxis().SetTitleSize(label_size)
    
    leg = ROOT.TLegend(legx1,legy1,legx2,legy2)
    leg.SetNColumns(2)
    leg.SetHeader('#bf{MicroBooNE Preliminary}              '+str(run1dataPOT+run2dataPOT+run3dataPOT)+' POT',"C")
    for h,l in zip(root_hists, mc_labels):
        leg.AddEntry(h,l)
    if systdir != "":
        leg.AddEntry(hmcerror, "Pred. uncertainty", "lf")
    leg.Draw()
    
    if plotlog:
        c.SetLogy()
    c.Update()
    #c.Draw()
    c.Print('plots/root_plots/'+plot_folder+'/mc_only/'+title+'.png')
    
    return h_sig, h_bkg

def MakeVarPlots(var_list, num_bins, folder_name, plot_folder_name, selection):
    #make and save data/mc hists for all variables in var_list with any cuts made before function call
    #var_list: array of variable names in input files (e.g. all_sp_scalars, load_varaibles, etc.)
    #num_bins: number of bins for all plots
    #folder_name: name of folder to save plots in
    if not os.path.exists('plots/'+folder_name):
        os.makedirs('plots/'+folder_name)
        
    plot_folder_temp = plot_folder_name
    plot_folder = plot_folder_name+'/'+folder_name
    if not os.path.exists('plots/root_plots/'+plot_folder):
        os.makedirs('plots/root_plots/'+plot_folder)
        os.makedirs('plots/root_plots/'+plot_folder+'/datamc')
        os.makedirs('plots/root_plots/'+plot_folder+'/data')
        os.makedirs('plots/root_plots/'+plot_folder+'/mc_only')
    for var_name in var_list:
        var = all_df[var_name].to_numpy()   
        var_sig = []
        var_bkg = []
        var_data = []

        for i in range(num_evts):
            if y[i]<4 or y[i]>100:
                var_sig.append(var[i])
            elif y[i]==13:
                var_data.append(var[i])
            elif y[i]>3 and y[i]!=13 and y[i]<100:
                var_bkg.append(var[i])  

        bin_width = ((1.0 + max(var))-min(var))/num_bins
        if (bin_width==0):
            print(var_name+" is empty")
        else:
            plot = MakeDataMCPlot(var_data,var_sig,var_bkg,bin_width,min(var),1.0 + max(var),var_name,var_name,"",0,0,0,selection)
            plot.savefig("plots/"+folder_name+"/"+var_name+".png", format='png',facecolor='white', transparent=False)
            plt.close()
        del var_sig
        del var_bkg
        del var_data
    
    plot_folder = plot_folder_temp

###
def MakeEffPurPlots(var_sig, var_bkg, bin_width, start_edge, end_edge, title, x_label, selection):
    
    #function to make effciency and purity plots for a variable, will use whatever cut value and part of chain 
    #comes before the call to the function
    #inputs:
        #var_sig/bkg: array of variable values for sig/bkg, array
        #bin_width: desired bin width (in variable units), float
        #start_edge: x-axis start value (inclusive), float or int
        #end_edge: x-axis end value (exclusive), float or int
        #title: title of plot, string
        #x_label: x-axis label, string
        
    selected_var_sig = []
    selected_var_bkg = []
    
    selected_w_sig = []
    selected_w_bkg = []
    
    selected_true_event_type_sig = []
    selected_true_event_type_bkg = []

    for i in range(0, len(e_sig)):
        if(PassSelection(selection, single_photon_numu_score_sig[i], single_photon_other_score_sig[i], single_photon_ncpi0_score_sig[i], single_photon_nue_score_sig[i], num_shw_sig[i], num_pro_sig[i], r_sig[i], s_sig[i], e_sig[i])):
            selected_var_sig.append(var_sig[i])
            selected_w_sig.append(weights_sig[i])
            selected_true_event_type_sig.append(true_event_type_sig[i])

    for i in range(0, len(e_bkg)):
        if(PassSelection(selection, single_photon_numu_score_bkg[i], single_photon_other_score_bkg[i], single_photon_ncpi0_score_bkg[i], single_photon_nue_score_bkg[i], num_shw_bkg[i], num_pro_bkg[i], r_bkg[i], s_bkg[i], e_bkg[i])):
            selected_var_bkg.append(var_bkg[i])
            selected_w_bkg.append(weights_bkg[i])
            selected_true_event_type_bkg.append(true_event_type_bkg[i])

    # picking out specific backgrounds from the selected events

    selected_ext_var = []
    selected_dirt_var = []
    selected_cos_var = []
    selected_outFV_var = []
    selected_numuCCpi0_var = []
    selected_numuCC_var = []
    selected_NCpi0_var = []
    selected_NC_var = []
    selected_nueCC_var = []
    selected_NCpi1g_var = []
    selected_NCdel_var = []
    selected_NCother_var = []
    selected_numuCC1g_var = []
    
    selected_ext_w = []
    selected_dirt_w = []
    selected_cos_w = []
    selected_outFV_w = []
    selected_numuCCpi0_w = []
    selected_numuCC_w = []
    selected_NCpi0_w = []
    selected_NC_w = []
    selected_nueCC_w = []
    selected_NCpi1g_w = []
    selected_NCdel_w = []
    selected_NCother_w = []
    selected_numuCC1g_w = []


    for i in range(len(selected_var_bkg)):
        if selected_true_event_type_bkg[i]==12:
            selected_ext_var.append(selected_var_bkg[i])
            selected_ext_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==11:
            selected_dirt_var.append(selected_var_bkg[i])
            selected_dirt_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==10:
            selected_cos_var.append(selected_var_bkg[i])
            selected_cos_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==9:
            selected_outFV_var.append(selected_var_bkg[i])
            selected_outFV_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==8: 
            selected_numuCCpi0_var.append(selected_var_bkg[i])
            selected_numuCCpi0_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==7:
            selected_numuCC_var.append(selected_var_bkg[i])
            selected_numuCC_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==6:
            selected_NCpi0_var.append(selected_var_bkg[i])
            selected_NCpi0_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==5:
            selected_NC_var.append(selected_var_bkg[i])
            selected_NC_w.append(selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==4: 
            selected_nueCC_var.append(selected_var_bkg[i])
            selected_nueCC_w.append(selected_w_bkg[i])

        else:
            print("There is an unknown additional background type")
            #print(selected_is_CC_bkg[i])
            print(selected_true_event_type_bkg[i])
            #print(selected_nu_Pdg_bkg[i])

    for i in range(len(selected_var_sig)):
        if selected_true_event_type_sig[i]==3: 
            selected_NCpi1g_var.append(selected_var_sig[i])
            selected_NCpi1g_w.append(selected_w_sig[i])
        elif selected_true_event_type_sig[i]==2: 
            selected_NCdel_var.append(selected_var_sig[i])
            selected_NCdel_w.append(selected_w_sig[i])
        elif selected_true_event_type_sig[i]==1: 
            selected_NCother_var.append(selected_var_sig[i])
            selected_NCother_w.append(selected_w_sig[i])
        elif selected_true_event_type_sig[i]==0: 
            selected_numuCC1g_var.append(selected_var_sig[i])
            selected_numuCC1g_w.append(selected_w_sig[i])
    
    bin_num = int((end_edge-start_edge)/bin_width)
    
    bin_edges = [(i * bin_width)+start_edge for i in range(bin_num+1)]
    bin_centers = [((i * bin_width)+start_edge)+(bin_width/2.)  for i in range(bin_num)]
    num_bins = len(bin_edges) - 1

    counts_sig_true, bin_edges, plot = plt.hist(var_sig, bins=bin_edges, weights=weights_sig[:])
    counts_bkg_true, bin_edges, plot = plt.hist(var_bkg, bins=bin_edges, weights=weights_bkg[:])

    counts_sig_true_unweighted, bin_edges, plot = plt.hist(var_sig, bins=bin_edges)
    counts_bkg_true_unweighted, bin_edges, plot = plt.hist(var_bkg, bins=bin_edges)

    counts_sig_sel_unweighted, bin_edges, plot = plt.hist(selected_var_sig,bins=bin_edges)
    counts_bkg_sel_unweighted, bin_edges, plot = plt.hist(selected_var_bkg,bins=bin_edges)
    

    plt.clf()
    plt.figure(dpi=100)
    counts_sig_sel, bin_edges, plot = plt.hist(selected_var_sig,bins=bin_edges,
                                               weights=selected_w_sig[:], label="signal",alpha=0.5)
    counts_bkg_sel, bin_edges, plot = plt.hist(selected_var_bkg,bins=bin_edges,
                                               weights=selected_w_bkg[:], label="background",alpha=0.5)
    
    selected_efficiency = np.sum(counts_sig_sel) / np.sum(counts_sig_true)
    selected_purity = np.sum(counts_sig_sel) / (np.sum(counts_sig_sel) + np.sum(counts_bkg_sel))


    plt.title("Cut value: " + str(cut_value) + ", purity: "+str(round(selected_purity, 3)) + 
                 ", sig eff: " + str(round(selected_efficiency, 3)), fontsize = title_size)
    plt.legend(prop={'size': 12})
    plt.xlabel(x_label)
    plt.show()

    S = counts_sig_sel
    S_true = counts_sig_true
    S_true_unweighted = counts_sig_true_unweighted
    S_unweighted = counts_sig_sel_unweighted # should be same as S

    B = counts_bkg_sel
    B_true = counts_bkg_true
    B_true_unweighted = counts_bkg_true_unweighted
    B_unweighted = counts_bkg_sel_unweighted 

    # binomial standard deviations
    sigma_S = [S_true[i] / S_true_unweighted[i] * 
               np.sqrt(S_unweighted[i] * (1. - S_unweighted[i] / S_true_unweighted[i])) 
                    if S_true_unweighted[i] != 0 else 1e6
                    for i in range(num_bins)]
    sigma_B = [B_true[i] / B_true_unweighted[i] * 
               np.sqrt(B_unweighted[i] * (1. - B_unweighted[i] / B_true_unweighted[i]))
                    if B_true_unweighted[i] != 0 else 1e6
                    for i in range(num_bins)]

    eff = [S[i] / S_true[i] for i in range(num_bins)]
    pur = [S[i] / (S[i] + B[i]) for i in range(num_bins)]

    eff_err = [sigma_S[i] / S_true[i] for i in range(num_bins)]

    # err of A/(A+B) is sqrt(B^2*sigma_A^2+A^2*sigma_B^2)/(A+B)^2
    pur_err = [np.sqrt((B[i] * sigma_S[i])**2 + (S[i] * sigma_B[i])**2) / (S[i] + B[i])**2 for i in range(num_bins)]

    for i in range(num_bins):
        if S_true[i] == 0:
            eff_err[i] = 100.
        if S_true[i] == 0 and B_true[i] == 0:
            pur_err[i] = 100.

    plt.figure(dpi=100)
    plt.errorbar(bin_centers, eff, yerr=eff_err, fmt='.', capsize=5., label = "efficiency")
    #plt.errorbar(bin_centers, pur, yerr=pur_err, fmt='.', capsize=5., label = "purity")
    #plt.ylim(-0.1, 1.1)
    #plt.legend(prop={'size': 12})
    plt.xlabel(x_label)
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs "+title, fontsize = title_size)
    plt.show()

    plt.figure(dpi=100)
    #plt.errorbar(bin_centers, eff, yerr=eff_err, fmt='.', capsize=5., label = "efficiency")
    plt.errorbar(bin_centers, pur, yerr=pur_err, fmt='.', capsize=5., label = "purity")
    #plt.ylim(-0.1, 1.1)
    #plt.legend(prop={'size': 12})
    plt.xlabel(x_label)
    plt.ylabel("Purity")
    plt.title("Purity vs "+title, fontsize = title_size)
    plt.show()

###
def Make2DPlot(varx_data, varx_sig, varx_bkg, vary_data, vary_sig, vary_bkg,
               bin_widthx, start_edgex, end_edgex, bin_widthy, start_edgey, end_edgey,
               title, x_label, y_label, event_types, selection):
    
    #function to make a 2D histogram plot for two variables, will use whatever cut value and part of chain 
    #comes before the call to the function
    #inputs:
        #varx/y_data/sig/bkg: array of variable values for data/sig/bkg for variable 1/2, array
        #bin_width: desired bin width (in variable units), float
        #start_edge: x-axis start value (inclusive), float or int
        #end_edge: x-axis end value (exclusive), float or int
        #title: title of plot, string
        #x_label: x-axis label, string
        #y_label: y-axis label, string
        #event_types: array of true_event_type numbers to plot
        
    bin_numx = int((end_edgex-start_edgex)/bin_widthx)
    
    bin_edgesx = [(i * bin_widthx)+start_edgex for i in range(bin_numx+1)]
    bin_centersx = [((i * bin_widthx)+start_edgex)+(bin_widthx/2.)  for i in range(bin_numx)]
    num_binsx = len(bin_edgesx) - 1
    
    bin_numy = int((end_edgey-start_edgey)/bin_widthy)
    
    bin_edgesy = [(i * bin_widthy)+start_edgey for i in range(bin_numy+1)]
    bin_centersy = [((i * bin_widthy)+start_edgey)+(bin_widthy/2.)  for i in range(bin_numy)]
    num_binsy = len(bin_edgesy) - 1
    
    h_data = ROOT.gROOT.FindObject("h_data")
    if h_data != None:
        h_data.Delete()
    
    h_ext = ROOT.gROOT.FindObject("h_ext")
    h_dirt = ROOT.gROOT.FindObject("h_dirt")
    h_cos = ROOT.gROOT.FindObject("h_cos")
    h_outFV = ROOT.gROOT.FindObject("h_outFV")
    h_numuCCpi0 = ROOT.gROOT.FindObject("h_numuCCpi0")
    h_numuCC = ROOT.gROOT.FindObject("h_numuCC")
    h_NCpi0 = ROOT.gROOT.FindObject("h_NCpi0")
    h_NC = ROOT.gROOT.FindObject("h_NC")
    h_nuepi0 = ROOT.gROOT.FindObject("h_nueCC")
    h_NCpi1g = ROOT.gROOT.FindObject("h_NCpi1g")
    h_NCdel = ROOT.gROOT.FindObject("h_NCdel")
    h_NCother = ROOT.gROOT.FindObject("h_NCother")
    h_numuCC1g = ROOT.gROOT.FindObject("h_numuCC1g")
    h_out1g = ROOT.gROOT.FindObject("h_out1g")
    if h_ext != None:
        h_ext.Delete()
    if h_dirt != None:
        h_dirt.Delete()
    if h_cos != None:
        h_cos.Delete()
    if h_outFV != None:
        h_outFV.Delete()
    if h_numuCCpi0 != None:
        h_numuCCpi0.Delete()
    if h_numuCC != None:
        h_numuCC.Delete()
    if h_NCpi0 != None:
        h_NCpi0.Delete()
    if h_NC != None:
        h_NC.Delete()
    if h_nuepi0 != None:
        h_nuepi0.Delete()
    if h_NCpi1g != None:
        h_NCpi1g.Delete()
    if h_NCdel != None:
        h_NCdel.Delete()
    if h_NCother != None:
        h_NCother.Delete()
    if h_numuCC1g != None:
        h_numuCC1g.Delete()
    if h_out1g != None:
        h_out1g.Delete()
    
    h_data = ROOT.TH2F('h_data', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_ext = ROOT.TH2F('h_ext', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_dirt = ROOT.TH2F('h_dirt', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_cos = ROOT.TH2F('h_cos', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_outFV = ROOT.TH2F('h_outFV', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_numuCCpi0 = ROOT.TH2F('h_numuCCpi0', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_numuCC = ROOT.TH2F('h_numuCC', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_NCpi0 = ROOT.TH2F('h_NCpi0', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_NC = ROOT.TH2F('h_NC', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_nueCC = ROOT.TH2F('h_nueCC', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_NCpi1g = ROOT.TH2F('h_NCpi1g', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_NCdel = ROOT.TH2F('h_NCdel', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_NCother = ROOT.TH2F('h_NCother', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_numuCC1g = ROOT.TH2F('h_numuCC1g', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_out1g = ROOT.TH2F('h_out1g', title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)

    
    selected_varx_sig = []
    selected_varx_bkg = []
    selected_varx_data = []
    
    selected_vary_sig = []
    selected_vary_bkg = []
    selected_vary_data = []
    
    selected_w_sig = []
    selected_w_bkg = []
    selected_w_data = []
    
    selected_true_event_type_sig = []
    selected_true_event_type_bkg = []
    selected_true_event_type_data = []

    for i in range(0, len(e_sig)):
        if(PassSelection(selection, single_photon_numu_score_sig[i], single_photon_other_score_sig[i], single_photon_ncpi0_score_sig[i], single_photon_nue_score_sig[i], num_shw_sig[i], num_pro_sig[i], r_sig[i], s_sig[i], e_sig[i])):
            selected_varx_sig.append(varx_sig[i])
            selected_vary_sig.append(vary_sig[i])
            selected_w_sig.append(weights_sig[i])
            selected_true_event_type_sig.append(true_event_type_sig[i])

    for i in range(0, len(e_bkg)):
        if(PassSelection(selection, single_photon_numu_score_bkg[i], single_photon_other_score_bkg[i], single_photon_ncpi0_score_bkg[i], single_photon_nue_score_bkg[i], num_shw_bkg[i], num_pro_bkg[i], r_bkg[i], s_bkg[i], e_bkg[i])):
            selected_varx_bkg.append(varx_bkg[i])
            selected_vary_bkg.append(vary_bkg[i])
            selected_w_bkg.append(weights_bkg[i])
            selected_true_event_type_bkg.append(true_event_type_bkg[i])

    for i in range(0, len(e_data)):
        if(PassSelection(selection, single_photon_numu_score_data[i], single_photon_other_score_data[i], single_photon_ncpi0_score_data[i], single_photon_nue_score_data[i], num_shw_data[i], num_pro_data[i], r_data[i], s_data[i], e_data[i])):
            selected_varx_data.append(varx_data[i])
            selected_vary_data.append(vary_data[i])
            selected_w_data.append(weights_data[i])
            selected_true_event_type_data.append(true_event_type_data[i])
            if 13 in event_types:
                h_data.Fill(varx_data[i],vary_data[i],weights_data[i])

    # picking out specific backgrounds from the selected events
    
    selected_data_varx = []
    selected_data_vary = []
    if 13 in event_types:
        selected_data_varx = selected_varx_data
        selected_data_vary = selected_vary_data

    selected_ext_varx = []
    selected_dirt_varx = []
    selected_cos_varx = []
    selected_outFV_varx = []
    selected_numuCCpi0_varx = []
    selected_numuCC_varx = []
    selected_NCpi0_varx = []
    selected_NC_varx = []
    selected_nueCC_varx = []
    selected_NCpi1g_varx = []
    selected_NCdel_varx = []
    selected_NCother_varx = []
    selected_numuCC1g_varx = []
    selected_out1g_varx = []
    
    selected_ext_vary = []
    selected_dirt_vary = []
    selected_cos_vary = []
    selected_outFV_vary = []
    selected_numuCCpi0_vary = []
    selected_numuCC_vary = []
    selected_NCpi0_vary = []
    selected_NC_vary = []
    selected_nueCC_vary = []
    selected_NCpi1g_vary = []
    selected_NCdel_vary = []
    selected_NCother_vary = []
    selected_numuCC1g_vary = []
    selected_out1g_vary = []
    
    selected_ext_w = []
    selected_dirt_w = []
    selected_cos_w = []
    selected_outFV_w = []
    selected_numuCCpi0_w = []
    selected_numuCC_w = []
    selected_NCpi0_w = []
    selected_NC_w = []
    selected_nueCC_w = []
    selected_NCpi1g_w = []
    selected_NCdel_w = []
    selected_NCother_w = []
    selected_numuCC1g_w = []
    selected_out1g_w = []


    for i in range(len(selected_varx_bkg)):
        if selected_true_event_type_bkg[i]==12 and 12 in event_types:
            selected_ext_varx.append(selected_varx_bkg[i])
            selected_ext_vary.append(selected_vary_bkg[i])
            selected_ext_w.append(selected_w_bkg[i])
            h_ext.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==11 and 11 in event_types:
            selected_dirt_varx.append(selected_varx_bkg[i])
            selected_dirt_vary.append(selected_vary_bkg[i])
            selected_dirt_w.append(selected_w_bkg[i])
            h_dirt.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==10 and 10 in event_types:
            selected_cos_varx.append(selected_varx_bkg[i])
            selected_cos_vary.append(selected_vary_bkg[i])
            selected_cos_w.append(selected_w_bkg[i])
            h_cos.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==9 and 9 in event_types:
            selected_outFV_varx.append(selected_varx_bkg[i])
            selected_outFV_vary.append(selected_vary_bkg[i])
            selected_outFV_w.append(selected_w_bkg[i])
            h_outFV.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==8 and 8 in event_types: 
            selected_numuCCpi0_varx.append(selected_varx_bkg[i])
            selected_numuCCpi0_vary.append(selected_vary_bkg[i])
            selected_numuCCpi0_w.append(selected_w_bkg[i])
            h_numuCCpi0.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==7 and 7 in event_types:
            selected_numuCC_varx.append(selected_varx_bkg[i])
            selected_numuCC_vary.append(selected_vary_bkg[i])
            selected_numuCC_w.append(selected_w_bkg[i])
            h_numuCC.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==6 and 6 in event_types:
            selected_NCpi0_varx.append(selected_varx_bkg[i])
            selected_NCpi0_vary.append(selected_vary_bkg[i])
            selected_NCpi0_w.append(selected_w_bkg[i])
            h_NCpi0.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==5 and 5 in event_types:
            selected_NC_varx.append(selected_varx_bkg[i])
            selected_NC_vary.append(selected_vary_bkg[i])
            selected_NC_w.append(selected_w_bkg[i])
            h_NC.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])
        elif selected_true_event_type_bkg[i]==4 and 4 in event_types: 
            selected_nueCC_varx.append(selected_varx_bkg[i])
            selected_nueCC_vary.append(selected_vary_bkg[i])
            selected_nueCC_w.append(selected_w_bkg[i])
            h_nueCC.Fill(selected_varx_bkg[i],selected_vary_bkg[i],selected_w_bkg[i])

        #else:
            #print("There is an unknown additional background type")
            #print(selected_is_CC_bkg[i])
            #print(selected_true_event_type_bkg[i])
            #print(selected_nu_Pdg_bkg[i])

    for i in range(len(selected_varx_sig)):
        if selected_true_event_type_sig[i]==3 and 3 in event_types: 
            selected_NCpi1g_varx.append(selected_varx_sig[i])
            selected_NCpi1g_vary.append(selected_vary_sig[i])
            selected_NCpi1g_w.append(selected_w_sig[i])
            h_NCpi1g.Fill(selected_varx_sig[i],selected_vary_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==2 and 2 in event_types: 
            selected_NCdel_varx.append(selected_varx_sig[i])
            selected_NCdel_vary.append(selected_vary_sig[i])
            selected_NCdel_w.append(selected_w_sig[i])
            h_NCdel.Fill(selected_varx_sig[i],selected_vary_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==1 and 1 in event_types: 
            selected_NCother_varx.append(selected_varx_sig[i])
            selected_NCother_vary.append(selected_vary_sig[i])
            selected_NCother_w.append(selected_w_sig[i])
            h_NCother.Fill(selected_varx_sig[i],selected_vary_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==0 and 0 in event_types: 
            selected_numuCC1g_varx.append(selected_varx_sig[i])
            selected_numuCC1g_vary.append(selected_vary_sig[i])
            selected_numuCC1g_w.append(selected_w_sig[i])
            h_numuCC1g.Fill(selected_varx_sig[i],selected_vary_sig[i],selected_w_sig[i])
        elif selected_true_event_type_sig[i]==111 and 111 in event_types: 
            selected_out1g_varx.append(selected_varx_sig[i])
            selected_out1g_vary.append(selected_vary_sig[i])
            selected_out1g_w.append(selected_w_sig[i])
            h_out1g.Fill(selected_varx_sig[i],selected_vary_sig[i],selected_w_sig[i])
            
            
    
    # make the plots
    
    #counts_sig_true, bin_edgesx, bin_edgesy, plot = plt.hist2d(varx_sig, vary_sig, weights=weights_sig[:])
    #counts_bkg_true, bin_edgesx, bin_edgesy, plot = plt.hist2d(varx_bkg, vary_bkg, weights=weights_bkg[:])

    #counts_sig_true_unweighted, bin_edgesx, bin_edgesy, plot = plt.hist2d(varx_sig, vary_sig)
    #counts_bkg_true_unweighted, bin_edgesx, bin_edgesy, plot = plt.hist2d(varx_bkg, vary_bkg)

    #counts_sig_sel_unweighted, bin_edgesx, bin_edgesy, plot = plt.hist2d(selected_varx_sig, selected_vary_sig)
    #counts_bkg_sel_unweighted, bin_edgesx, bin_edgesy, plot = plt.hist2d(selected_varx_bkg, selected_vary_bkg)

    #counts_sig_sel, bin_edgesx, bin_edgesy, plot = plt.hist2d(selected_varx_sig, selected_vary_sig, weights=selected_w_sig[:])
    #counts_bkg_sel, bin_edgesx, bin_edgesy, plot = plt.hist2d(selected_varx_bkg, selected_vary_bkg, weights=selected_w_bkg[:])
    
    

    #plt.clf()
    #plt.figure(dpi=100)

    pred_varx = [selected_numuCC1g_varx, selected_NCother_varx, selected_NCdel_varx, selected_NCpi1g_varx, 
                 selected_nueCC_varx,
            selected_NC_varx, selected_NCpi0_varx, selected_numuCC_varx, selected_numuCCpi0_varx, 
                 selected_outFV_varx,
            selected_cos_varx, selected_dirt_varx, selected_ext_varx, selected_data_varx]
    
    pred_vary = [selected_numuCC1g_vary, selected_NCother_vary, selected_NCdel_vary, selected_NCpi1g_vary, 
                 selected_nueCC_vary,
            selected_NC_vary, selected_NCpi0_vary, selected_numuCC_vary, selected_numuCCpi0_vary, 
                 selected_outFV_vary,
            selected_cos_vary, selected_dirt_vary, selected_ext_vary, selected_data_vary]
    
    selected_varx_all = selected_numuCC1g_varx+selected_NCother_varx+selected_NCdel_varx+selected_NCpi1g_varx+selected_nueCC_varx+selected_NC_varx+selected_NCpi0_varx+selected_numuCC_varx+selected_numuCCpi0_varx+selected_outFV_varx+selected_cos_varx+selected_dirt_varx+selected_ext_varx+selected_data_varx
        
    selected_vary_all = selected_numuCC1g_vary+selected_NCother_vary+selected_NCdel_vary+selected_NCpi1g_vary+selected_nueCC_vary+selected_NC_vary+selected_NCpi0_vary+selected_numuCC_vary+selected_numuCCpi0_vary+selected_outFV_vary+selected_cos_vary+selected_dirt_vary+selected_ext_vary+selected_data_vary
        
    

    #print(selected_varx_all)

   # mc_var_hist,mc_bins_varx, mc_bins_vary, patches_var = plt.hist2d(selected_varx_all, selected_vary_all, bins=[bin_edgesx, bin_edgesy], 
   #                                                  alpha=0.7)
                                                     #, weights=selected_w_bkg)#mc_weights,  
                                                     #color=colors+[ROOT.gROOT.GetColor(kBlack).AsHexString()], 
                                                     #range=[[start_edgex,end_edgex], [start_edgey,end_edgey]], 
                                                     #label=mc_labels+["Data"])


  #  plt.title(title, fontsize = title_size)
    #plt.titlesize(title_size)
    #handles, labels = plt.get_legend_handles_labels()
    #plt.legend(reversed(handles), reversed(labels), prop={'size': 12},ncol=2,
    #           title = 
    #           '{} POT                      Stat. Uncert. Only\n$\Sigma$Data/$\Sigma$(MC+EXT)={}'.format(run1dataPOT+run3dataPOT,
      #                                                                                                   sum_data_mc_ratio))
  #  plt.xlabel(x_label)
  #  plt.ylabel(y_label)
    
  # kPink-8).AsHexString(), 
  #        ROOT.gROOT.GetColor(kPink-6).AsHexString(), ROOT.gROOT.GetColor(kPink+5).AsHexString(), 
  #        ROOT.gROOT.GetColor(kGreen+1).AsHexString(), ROOT.gROOT.GetColor(kGreen+1).AsHexString(), 
  #        ROOT.gROOT.GetColor(38).AsHexString(), ROOT.gROOT.GetColor(kAzure+6).AsHexString(), 
  #        ROOT.gROOT.GetColor(30).AsHexString(), ROOT.gROOT.GetColor(kOrange+1).AsHexString(), 
  #        ROOT.gROOT.GetColor(kGray+2).AsHexString(), ROOT.gROOT.GetColor(kGray).AsHexString(), 
  #        ROOT.gROOT.GetColor(kGray).AsHexString()]
    
    c = ROOT.TCanvas(title,title,800,600)
    #c.cd()
    
    h_stack = ROOT.gROOT.FindObject("h_stack")
    if h_stack != None:
        h_stack.Delete()
        
    h_stack = ROOT.TH2F("h_stack",title, bin_numx, start_edgex, end_edgex, bin_numy, start_edgey, end_edgey)
    h_stack.GetXaxis().SetTitle(x_label)
    h_stack.GetYaxis().SetTitle(y_label)
    h_stack.GetXaxis().SetLabelSize(label_size)
    h_stack.GetYaxis().SetLabelSize(label_size)

    h_out1g.Draw()
    h_out1g.GetXaxis().SetTitle(x_label)
    h_out1g.GetYaxis().SetTitle(y_label)
    h_out1g.GetXaxis().SetLabelSize(label_size)
    h_out1g.GetYaxis().SetLabelSize(label_size)
    h_out1g.SetMarkerColor(kPink+1)
    h_stack.Add(h_out1g)
    
    h_numuCC1g.Draw()
    h_numuCC1g.GetXaxis().SetTitle(x_label)
    h_numuCC1g.GetYaxis().SetTitle(y_label)
    h_numuCC1g.GetXaxis().SetLabelSize(label_size)
    h_numuCC1g.GetYaxis().SetLabelSize(label_size)
    h_numuCC1g.SetMarkerColor(kPink-7)
    h_stack.Add(h_numuCC1g)
    
    h_NCother.Draw("same")
    h_NCother.GetXaxis().SetTitle(x_label)
    h_NCother.GetYaxis().SetTitle(y_label)
    h_NCother.GetXaxis().SetLabelSize(label_size)
    h_NCother.GetYaxis().SetLabelSize(label_size)
    h_NCother.SetMarkerColor(kPink-8)
    h_stack.Add(h_NCother)
    
    h_NCdel.Draw("same")
    h_NCdel.GetXaxis().SetTitle(x_label)
    h_NCdel.GetYaxis().SetTitle(y_label)
    h_NCdel.GetXaxis().SetLabelSize(label_size)
    h_NCdel.GetYaxis().SetLabelSize(label_size)
    h_NCdel.SetMarkerColor(kPink-6)
    h_stack.Add(h_NCdel)
    
    h_NCpi1g.Draw("same")
    h_NCpi1g.GetXaxis().SetTitle(x_label)
    h_NCpi1g.GetYaxis().SetTitle(y_label)
    h_NCpi1g.GetXaxis().SetLabelSize(label_size)
    h_NCpi1g.GetYaxis().SetLabelSize(label_size)
    h_NCpi1g.SetMarkerColor(kPink+5)
    h_stack.Add(h_NCpi1g)
    
    h_nueCC.Draw("same")
    h_nueCC.GetXaxis().SetTitle(x_label)
    h_nueCC.GetYaxis().SetTitle(y_label)
    h_nueCC.GetXaxis().SetLabelSize(label_size)
    h_nueCC.GetYaxis().SetLabelSize(label_size)
    h_nueCC.SetMarkerColor(kGreen+1)
    h_stack.Add(h_nueCC)
    
    h_NC.Draw("same")
    h_NC.GetXaxis().SetTitle(x_label)
    h_NC.GetYaxis().SetTitle(y_label)
    h_NC.GetXaxis().SetLabelSize(label_size)
    h_NC.GetYaxis().SetLabelSize(label_size)
    h_NC.SetMarkerColor(kGreen+1)
    h_stack.Add(h_NC)
    
    h_NCpi0.Draw("same")
    h_NCpi0.GetXaxis().SetTitle(x_label)
    h_NCpi0.GetYaxis().SetTitle(y_label)
    h_NCpi0.GetXaxis().SetLabelSize(label_size)
    h_NCpi0.GetYaxis().SetLabelSize(label_size)
    h_NCpi0.SetMarkerColor(38)
    h_stack.Add(h_NCpi0)
    
    h_numuCC.Draw("same")
    h_numuCC.GetXaxis().SetTitle(x_label)
    h_numuCC.GetYaxis().SetTitle(y_label)
    h_numuCC.GetXaxis().SetLabelSize(label_size)
    h_numuCC.GetYaxis().SetLabelSize(label_size)
    h_numuCC.SetMarkerColor(kAzure+6)
    h_stack.Add(h_numuCC)
    
    h_numuCCpi0.Draw("same")
    h_numuCCpi0.GetXaxis().SetTitle(x_label)
    h_numuCCpi0.GetYaxis().SetTitle(y_label)
    h_numuCCpi0.GetXaxis().SetLabelSize(label_size)
    h_numuCCpi0.GetYaxis().SetLabelSize(label_size)
    h_numuCCpi0.SetMarkerColor(30)
    h_stack.Add(h_numuCCpi0)
    
    h_outFV.Draw("same")
    h_outFV.GetXaxis().SetTitle(x_label)
    h_outFV.GetYaxis().SetTitle(y_label)
    h_outFV.GetXaxis().SetLabelSize(label_size)
    h_outFV.GetYaxis().SetLabelSize(label_size)
    h_outFV.SetMarkerColor(kOrange+1)
    h_stack.Add(h_outFV)
    
    h_cos.Draw("same")
    h_cos.GetXaxis().SetTitle(x_label)
    h_cos.GetYaxis().SetTitle(y_label)
    h_cos.GetXaxis().SetLabelSize(label_size)
    h_cos.GetYaxis().SetLabelSize(label_size)
    h_cos.SetMarkerColor(kGray+2)
    h_stack.Add(h_cos)
    
    h_dirt.Draw("same")
    h_dirt.GetXaxis().SetTitle(x_label)
    h_dirt.GetYaxis().SetTitle(y_label)
    h_dirt.GetXaxis().SetLabelSize(label_size)
    h_dirt.GetYaxis().SetLabelSize(label_size)
    h_dirt.SetMarkerColor(kGray)
    h_stack.Add(h_dirt)
    
    h_ext.Draw("same")
    h_ext.GetXaxis().SetTitle(x_label)
    h_ext.GetYaxis().SetTitle(y_label)
    h_ext.GetXaxis().SetLabelSize(label_size)
    h_ext.GetYaxis().SetLabelSize(label_size)
    h_ext.SetMarkerColor(kGray+3)
    h_stack.Add(h_ext)
    
    h_data.Draw("same")
    h_data.GetXaxis().SetTitle(x_label)
    h_data.GetYaxis().SetTitle(y_label)
    h_data.GetXaxis().SetLabelSize(label_size)
    h_data.GetYaxis().SetLabelSize(label_size)
    h_data.SetMarkerColor(kBlack)
    h_data.SetMarkerStyle(34)
    h_stack.Add(h_data)
    
    h_stack.Draw("colz")
    c.Update()
    #c.Draw()
    c.Print('plots/root_plots/'+plot_folder+'/plots_2d/'+title+'.png')
    
    
    
    return h_stack
    #return c,h_data,h_ext,h_dirt,h_cos,h_outFV,h_numuCCpi0,h_numuCC,h_NCpi0,h_NC,h_nueCC,h_NCpi1g,h_NCdel,h_NCother,h_numuCC1g


### Variables to load
# This is a list of the scalar variables saved for the numu tagger

numu3_var = ["numu_cc_flag_3",
"numu_cc_3_particle_type",
"numu_cc_3_max_length",
"numu_cc_3_track_length",#numu_cc_3_acc_track_length'
"numu_cc_3_max_length_all",
"numu_cc_3_max_muon_length",
"numu_cc_3_n_daughter_tracks",
"numu_cc_3_n_daughter_all"]

cosmict24_var = ["cosmict_flag_2",
"cosmict_2_filled",
"cosmict_2_particle_type",
"cosmict_2_n_muon_tracks",
"cosmict_2_total_shower_length",
"cosmict_2_flag_inside",
"cosmict_2_angle_beam",
"cosmict_2_flag_dir_weak",
"cosmict_2_dQ_dx_end",
"cosmict_2_dQ_dx_front",
"cosmict_2_theta",
"cosmict_2_phi",
"cosmict_2_valid_tracks",
"cosmict_flag_4", 
"cosmict_4_filled",
"cosmict_4_flag_inside",
"cosmict_4_angle_beam",
"cosmict_4_connected_showers"]

cosmict35_var = ["cosmict_flag_3",
"cosmict_3_filled",
"cosmict_3_flag_inside",
"cosmict_3_angle_beam",
"cosmict_3_flag_dir_weak",
"cosmict_3_dQ_dx_end",
"cosmict_3_dQ_dx_front",
"cosmict_3_theta",
"cosmict_3_phi",
"cosmict_3_valid_tracks",
"cosmict_flag_5",  
"cosmict_5_filled",
"cosmict_5_flag_inside",
"cosmict_5_angle_beam",
"cosmict_5_connected_showers"]

cosmict6_var = ["cosmict_flag_6", 
"cosmict_6_filled",
"cosmict_6_flag_dir_weak",
"cosmict_6_flag_inside",
"cosmict_6_angle"]

cosmict7_var = ["cosmict_flag_7",
"cosmict_7_filled",
"cosmict_7_flag_sec",
"cosmict_7_n_muon_tracks",
"cosmict_7_total_shower_length",
"cosmict_7_flag_inside",
"cosmict_7_angle_beam",
"cosmict_7_flag_dir_weak",
"cosmict_7_dQ_dx_end",
"cosmict_7_dQ_dx_front",
"cosmict_7_theta",
"cosmict_7_phi"]

cosmict8_var = ["cosmict_flag_8", 
"cosmict_8_filled",
"cosmict_8_flag_out",
"cosmict_8_muon_length",
"cosmict_8_acc_length"]

cosmict9_var = ["cosmict_flag_9",
"cosmic_flag",
"cosmic_filled"]

overall_var = ["cosmict_flag",
"numu_cc_flag"]

all_numu_scalars = []
all_numu_scalars += numu3_var
all_numu_scalars += cosmict24_var
all_numu_scalars += cosmict35_var
all_numu_scalars += cosmict6_var
all_numu_scalars += cosmict7_var
all_numu_scalars += cosmict8_var
all_numu_scalars += cosmict9_var
all_numu_scalars += overall_var

all_numu_scalars += ["cosmict_flag_1"]#, "match_isFC"] #"kine_reco_Enu",

#remaining_vars = []

numu_bkg_vars = []
numu_bkg_vars += numu3_var

other_bkg_vars = []
#other_bkg_vars += ["numu_cc_flag"]
other_bkg_vars += cosmict24_var + cosmict35_var + cosmict6_var + cosmict7_var + cosmict8_var + cosmict9_var + overall_var
 

ncpi0_bkg_vars = []

nue_bkg_vars = []

bdt_variables = []
bdt_variables+=overall_var
#bdt_variables+=all_numu_scalars
#bdt_variables+=cosmict24_var + cosmict35_var + cosmict6_var + cosmict7_var + cosmict8_var + cosmict9_var + overall_var

# this is a list of the vector variables saved for the numu tagger (currently not using these for training, that would be more complicated)

var_numu1 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'numu_cc_flag_1',
             'numu_cc_1_particle_type',
             'numu_cc_1_length',
             'numu_cc_1_medium_dQ_dx',
             'numu_cc_1_dQ_dx_cut',
             'numu_cc_1_direct_length',
             'numu_cc_1_n_daughter_tracks',
             'numu_cc_1_n_daughter_all']
var_numu2 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'numu_cc_flag_2',
             'numu_cc_2_length',
             'numu_cc_2_total_length',
             'numu_cc_2_n_daughter_tracks',
             'numu_cc_2_n_daughter_all']
var_cos10 = [#'weight',
             #'numu_cc_flag',
             #'cosmict_flag_1',
             #'cosmict_flag_10',
             #'cosmict_10_flag_inside',
             'cosmict_10_vtx_z',
             'cosmict_10_flag_shower',
             'cosmict_10_flag_dir_weak',
             'cosmict_10_angle_beam',
             'cosmict_10_length']

all_numu_vectors = []
all_numu_vectors += var_numu1
all_numu_vectors += var_numu2
all_numu_vectors += var_cos10

#bdt_variables+=all_numu_vectors

taggerCMEAMC_var = ["cme_mu_energy","cme_energy","cme_mu_length","cme_length",
                "cme_angle_beam","anc_angle","anc_max_angle","anc_max_length",
                "anc_acc_forward_length","anc_acc_backward_length","anc_acc_forward_length1",
                "anc_shower_main_length","anc_shower_total_length","anc_flag_main_outside"]

taggerGAP_var = ["gap_flag_prolong_u","gap_flag_prolong_v","gap_flag_prolong_w","gap_flag_parallel",
                 "gap_n_points","gap_n_bad","gap_energy","gap_num_valid_tracks","gap_flag_single_shower"]

taggerHOL_var = ["hol_1_n_valid_tracks","hol_1_min_angle","hol_1_energy","hol_1_flag_all_shower","hol_1_min_length",
               "hol_2_min_angle","hol_2_medium_dQ_dx","hol_2_ncount","lol_3_angle_beam","lol_3_n_valid_tracks",
               "lol_3_min_angle","lol_3_vtx_n_segs","lol_3_shower_main_length","lol_3_n_out","lol_3_n_sum"]

taggerMGOMGT_var = ["mgo_energy","mgo_max_energy","mgo_total_energy","mgo_n_showers","mgo_max_energy_1",
                    "mgo_max_energy_2","mgo_total_other_energy","mgo_n_total_showers","mgo_total_other_energy_1",
                   "mgt_flag_single_shower","mgt_max_energy","mgt_total_other_energy","mgt_max_energy_1",
                   "mgt_e_indirect_max_energy","mgt_e_direct_max_energy","mgt_n_direct_showers",
                    "mgt_e_direct_total_energy","mgt_flag_indirect_max_pio","mgt_e_indirect_total_energy"]

taggerMIPQUALITY_var = ["mip_quality_energy","mip_quality_overlap","mip_quality_n_showers","mip_quality_n_tracks",
                        "mip_quality_flag_inside_pi0","mip_quality_n_pi0_showers","mip_quality_shortest_length",
                        "mip_quality_acc_length","mip_quality_shortest_angle","mip_quality_flag_proton"]

taggerBR1_var = ["br1_1_shower_type","br1_1_vtx_n_segs","br1_1_energy","br1_1_n_segs","br1_1_flag_sg_topology",
                 "br1_1_flag_sg_trajectory","br1_1_sg_length","br1_2_n_connected","br1_2_max_length",
                "br1_2_n_connected_1","br1_2_n_shower_segs","br1_2_max_length_ratio","br1_2_shower_length",
                 "br1_3_n_connected_p","br1_3_max_length_p","br1_3_n_shower_main_segs"]

taggerBR3_var = ["br3_1_energy","br3_1_n_shower_segments","br3_1_sg_flag_trajectory","br3_1_sg_direct_length",
                "br3_1_sg_length","br3_1_total_main_length","br3_1_total_length","br3_1_iso_angle",
                 "br3_1_sg_flag_topology","br3_2_n_ele","br3_2_n_other","br3_2_other_fid","br3_4_acc_length",
                 "br3_4_total_length","br3_7_min_angle","br3_8_max_dQ_dx","br3_8_n_main_segs"]

taggerBR4TRE_var = ["br4_1_shower_main_length","br4_1_shower_total_length","br4_1_min_dis","br4_1_energy",
                    "br4_1_flag_avoid_muon_check","br4_1_n_vtx_segs","br4_1_n_main_segs","br4_2_ratio_45",
                   "br4_2_ratio_35","br4_2_ratio_25","br4_2_ratio_15","br4_2_ratio1_45","br4_2_ratio1_35",
                   "br4_2_ratio1_25","br4_2_ratio1_15","br4_2_iso_angle","br4_2_iso_angle1","br4_2_angle",
                   "tro_3_stem_length","tro_3_n_muon_segs"]

taggerVIS1_var = ["vis_1_n_vtx_segs","vis_1_energy","vis_1_num_good_tracks","vis_1_max_angle",
                  "vis_1_max_shower_angle","vis_1_tmp_length1","vis_1_tmp_length2"]

taggerVIS2_var = ["vis_2_n_vtx_segs","vis_2_min_angle","vis_2_min_weak_track","vis_2_angle_beam","vis_2_min_angle1",
                 "vis_2_iso_angle1","vis_2_min_medium_dQ_dx","vis_2_min_length","vis_2_sg_length","vis_2_max_angle",
                 "vis_2_max_weak_track"]

taggerPI01_var = ["pio_1_mass","pio_1_pio_type","pio_1_energy_1","pio_1_energy_2","pio_1_dis_1","pio_1_dis_2","pio_mip_id"]

taggerSTEMDIRBR2_var = ["stem_dir_flag_single_shower","stem_dir_angle","stem_dir_energy","stem_dir_angle1",
                        "stem_dir_angle2","stem_dir_angle3","stem_dir_ratio","br2_num_valid_tracks",
                        "br2_n_shower_main_segs","br2_max_angle","br2_sg_length","br2_flag_sg_trajectory"]

taggerSTLLEMBRM_var = ["stem_len_energy","stem_len_length","stem_len_flag_avoid_muon_check","stem_len_num_daughters",
                      "stem_len_daughter_length","brm_n_mu_segs","brm_Ep","brm_acc_length","brm_shower_total_length",
                      "brm_connected_length","brm_n_size","brm_acc_direct_length","brm_n_shower_main_segs",
                       "brm_n_mu_main","lem_shower_main_length","lem_n_3seg","lem_e_charge","lem_e_dQdx",
                       "lem_shower_num_main_segs"]

taggerSTWSPT_var = ["stw_1_energy","stw_1_dis","stw_1_dQ_dx","stw_1_flag_single_shower","stw_1_n_pi0",
                    "stw_1_num_valid_tracks","spt_shower_main_length","spt_shower_total_length","spt_angle_beam",
                    "spt_angle_vertical","spt_max_dQ_dx","spt_angle_beam_1","spt_angle_drift","spt_angle_drift_1",
                    "spt_num_valid_tracks","spt_n_vtx_segs","spt_max_length"]

taggerMIP_var = ["mip_energy","mip_n_end_reduction","mip_n_first_mip","mip_n_first_non_mip","mip_n_first_non_mip_1",
                "mip_n_first_non_mip_2","mip_vec_dQ_dx_0","mip_vec_dQ_dx_1","mip_max_dQ_dx_sample",
                "mip_n_below_threshold","mip_n_below_zero","mip_n_lowest","mip_n_highest","mip_lowest_dQ_dx",
                 "mip_highest_dQ_dx","mip_medium_dQ_dx","mip_stem_length","mip_length_main","mip_length_total",
                 "mip_angle_beam","mip_iso_angle","mip_n_vertex","mip_n_good_tracks","mip_E_indirect_max_energy",
                 "mip_flag_all_above","mip_min_dQ_dx_5","mip_n_other_vertex","mip_n_stem_size",
                 "mip_flag_stem_trajectory","mip_min_dis"]

taggerAdditional_Var = ["mip_vec_dQ_dx_2","mip_vec_dQ_dx_3","mip_vec_dQ_dx_4","mip_vec_dQ_dx_5","mip_vec_dQ_dx_6",
                        "mip_vec_dQ_dx_7","mip_vec_dQ_dx_8","mip_vec_dQ_dx_9","mip_vec_dQ_dx_10","mip_vec_dQ_dx_11",
                        "mip_vec_dQ_dx_12","mip_vec_dQ_dx_13","mip_vec_dQ_dx_14","mip_vec_dQ_dx_15",
                        "mip_vec_dQ_dx_16","mip_vec_dQ_dx_17","mip_vec_dQ_dx_18","mip_vec_dQ_dx_19"]


all_nue_scalars = []
all_nue_scalars += taggerCMEAMC_var
all_nue_scalars += taggerGAP_var
all_nue_scalars += taggerHOL_var
all_nue_scalars += taggerMGOMGT_var
all_nue_scalars += taggerMIPQUALITY_var
all_nue_scalars += taggerBR1_var
all_nue_scalars += taggerBR3_var
all_nue_scalars += taggerBR4TRE_var
all_nue_scalars += taggerSTEMDIRBR2_var
all_nue_scalars += taggerSTLLEMBRM_var
all_nue_scalars += taggerSTWSPT_var
all_nue_scalars += taggerMIP_var
all_nue_scalars += taggerVIS1_var
all_nue_scalars += taggerVIS2_var
all_nue_scalars += taggerPI01_var
all_nue_scalars += taggerAdditional_Var

# removing this line, these variables are already in all_numu_scalars
#all_nue_scalars += ["kine_reco_Enu", "match_isFC"]

#bdt_variables+=all_nue_scalars

taggerHOL_var = [#"shw_sp_hol_flag",
                 "shw_sp_hol_1_n_valid_tracks",
                 #"shw_sp_hol_1_min_angle","shw_sp_hol_1_energy",
                 "shw_sp_hol_1_flag_all_shower", "shw_sp_hol_1_min_length", 
                 #"shw_sp_hol_2_min_angle",
                 "shw_sp_hol_2_medium_dQ_dx","shw_sp_hol_2_ncount",
                 #"shw_sp_lol_3_angle_beam","shw_sp_lol_flag", 
                 "shw_sp_lol_3_n_valid_tracks", 
                 #"shw_sp_lol_3_min_angle",
                 "shw_sp_lol_3_vtx_n_segs",
                 "shw_sp_lol_3_shower_main_length","shw_sp_lol_3_n_out","shw_sp_lol_3_n_sum"]

taggerBR1_var = [#"shw_sp_br1_flag",
                 "shw_sp_br1_1_shower_type","shw_sp_br1_1_vtx_n_segs",
                 #"shw_sp_br1_1_energy",
                 "shw_sp_br1_1_n_segs",
                 #"shw_sp_br1_1_flag_sg_topology", "shw_sp_br1_1_flag_sg_trajectory",
                 "shw_sp_br1_1_sg_length",
                 "shw_sp_br1_2_n_connected","shw_sp_br1_2_max_length", "shw_sp_br1_2_n_connected_1",
                 "shw_sp_br1_2_n_shower_segs","shw_sp_br1_2_max_length_ratio","shw_sp_br1_2_shower_length",
                 "shw_sp_br1_3_n_connected_p","shw_sp_br1_3_max_length_p","shw_sp_br1_3_n_shower_main_segs"]

taggerBR3_var = [#"shw_sp_br3_flag",
                #"shw_sp_br3_1_energy",
                 "shw_sp_br3_1_n_shower_segments",
                #"shw_sp_br3_1_sg_flag_trajectory",
                 "shw_sp_br3_1_sg_direct_length", "shw_sp_br3_1_sg_length","shw_sp_br3_1_total_main_length",
                 "shw_sp_br3_1_total_length",
                #"shw_sp_br3_1_iso_angle", "shw_sp_br3_1_sg_flag_topology",
                 "shw_sp_br3_2_n_ele","shw_sp_br3_2_n_other","shw_sp_br3_2_other_fid","shw_sp_br3_4_acc_length",
                 "shw_sp_br3_4_total_length",
                #"shw_sp_br3_7_min_angle",
                "shw_sp_br3_8_max_dQ_dx", 
                 "shw_sp_br3_8_n_main_segs"]

taggerBR4_var = [#"shw_sp_br4_flag",
                "shw_sp_br4_1_shower_main_length","shw_sp_br4_1_shower_total_length","shw_sp_br4_1_min_dis",
                 #"shw_sp_br4_1_energy",
                 "shw_sp_br4_1_flag_avoid_muon_check",
                 "shw_sp_br4_1_n_vtx_segs",
                 "shw_sp_br4_1_n_main_segs","shw_sp_br4_2_ratio_45", "shw_sp_br4_2_ratio_35","shw_sp_br4_2_ratio_25",
                 "shw_sp_br4_2_ratio_15","shw_sp_br4_2_ratio1_45","shw_sp_br4_2_ratio1_35", 
                 "shw_sp_br4_2_ratio1_25","shw_sp_br4_2_ratio1_15",
                 #"shw_sp_br4_2_iso_angle",
                 #"shw_sp_br4_2_iso_angle1","shw_sp_br4_2_angle"
                ]

taggerPI01_var = ["shw_sp_pio_1_mass","shw_sp_pio_1_pio_type","shw_sp_pio_1_energy_1","shw_sp_pio_1_energy_2",
                  "shw_sp_pio_1_dis_1","shw_sp_pio_1_dis_2", "shw_sp_pio_mip_id", #"shw_sp_pio_flag", 
                  "shw_sp_pio_flag_pio"]

taggerBR2_var = [#"shw_sp_br2_flag",
                 "shw_sp_br2_num_valid_tracks", "shw_sp_br2_n_shower_main_segs",
                 #"shw_sp_br2_max_angle",
                 "shw_sp_br2_sg_length"]#, "shw_sp_br2_flag_sg_trajectory"]

taggerLEM_var = [#"shw_sp_lem_flag",
                 "shw_sp_lem_shower_main_length","shw_sp_lem_n_3seg","shw_sp_lem_e_charge",
                 "shw_sp_lem_e_dQdx","shw_sp_lem_shower_num_main_segs"]

taggerMIP_var = [#"shw_sp_energy","shw_sp_flag",
                 "shw_sp_max_dQ_dx_sample",
                 "shw_sp_n_below_threshold","shw_sp_n_below_zero","shw_sp_n_lowest","shw_sp_n_highest",
                 "shw_sp_lowest_dQ_dx",
                 "shw_sp_highest_dQ_dx","shw_sp_medium_dQ_dx","shw_sp_stem_length","shw_sp_length_main",
                 "shw_sp_length_total",
                 #"shw_sp_angle_beam","shw_sp_iso_angle",
                 "shw_sp_n_vertex","shw_sp_n_good_tracks",
                 "shw_sp_E_indirect_max_energy",
                 "shw_sp_flag_all_above","shw_sp_min_dQ_dx_5","shw_sp_n_other_vertex","shw_sp_n_stem_size",
                 #"shw_sp_flag_stem_trajectory",
                 "shw_sp_min_dis", "shw_sp_vec_mean_dedx"]

taggerMIPdqdx_var = ["shw_sp_vec_dQ_dx_0","shw_sp_vec_dQ_dx_1","shw_sp_vec_dQ_dx_2","shw_sp_vec_dQ_dx_3",
                     "shw_sp_vec_dQ_dx_4","shw_sp_vec_dQ_dx_5","shw_sp_vec_dQ_dx_6","shw_sp_vec_dQ_dx_7",
                     "shw_sp_vec_dQ_dx_8","shw_sp_vec_dQ_dx_9","shw_sp_vec_dQ_dx_10","shw_sp_vec_dQ_dx_11",
                     "shw_sp_vec_dQ_dx_12","shw_sp_vec_dQ_dx_13","shw_sp_vec_dQ_dx_14","shw_sp_vec_dQ_dx_15",
                     "shw_sp_vec_dQ_dx_16","shw_sp_vec_dQ_dx_17","shw_sp_vec_dQ_dx_18","shw_sp_vec_dQ_dx_19",
                     "shw_sp_vec_median_dedx"]

taggerAdditional_var = ["shw_sp_proton_length_1", "shw_sp_proton_dqdx_1", "shw_sp_proton_energy_1",
                        "shw_sp_proton_length_2", "shw_sp_proton_dqdx_2", "shw_sp_proton_energy_2",
                        "shw_sp_n_good_showers", "shw_sp_n_20mev_showers", "shw_sp_n_br1_showers",
                        "shw_sp_n_br2_showers", "shw_sp_n_br3_showers", "shw_sp_n_br4_showers",
                        "shw_sp_n_20br1_showers", 
                        "shw_sp_shw_vtx_dis", "shw_sp_max_shw_dis"]

taggerNumTrk_var = ["shw_sp_num_mip_tracks","shw_sp_num_muons", "shw_sp_num_pions","shw_sp_num_protons"]


all_sp_scalars = []
all_sp_scalars += taggerHOL_var
all_sp_scalars += taggerBR1_var
all_sp_scalars += taggerBR3_var
all_sp_scalars += taggerBR4_var
all_sp_scalars += taggerBR2_var
all_sp_scalars += taggerLEM_var
all_sp_scalars += taggerMIP_var
all_sp_scalars += taggerMIPdqdx_var
all_sp_scalars += taggerPI01_var
all_sp_scalars += taggerAdditional_var
all_sp_scalars += taggerNumTrk_var
      
# these variables are also in all_numu_scalars
#all_sp_scalars += ["match_isFC"]

#numu_bkg_vars += taggerHOL_var
numu_bkg_vars += ["shw_sp_br1_1_shower_type","shw_sp_br1_1_vtx_n_segs","shw_sp_br1_1_n_segs",
                  "shw_sp_br1_1_sg_length", "shw_sp_br1_2_n_connected","shw_sp_br1_2_max_length", 
                  "shw_sp_br1_2_n_connected_1", "shw_sp_br1_2_n_shower_segs","shw_sp_br1_2_max_length_ratio",
                  "shw_sp_br1_2_shower_length", "shw_sp_br1_3_max_length_p","shw_sp_br1_3_n_shower_main_segs"]
numu_bkg_vars += taggerBR3_var
#numu_bkg_vars += taggerBR4_var
numu_bkg_vars += taggerLEM_var
numu_bkg_vars += taggerMIP_var
numu_bkg_vars += ["shw_sp_proton_length_1", "shw_sp_proton_dqdx_1", "shw_sp_proton_energy_1",
                  "shw_sp_proton_length_2", "shw_sp_proton_dqdx_2", "shw_sp_proton_energy_2",
                  "shw_sp_n_good_showers", "shw_sp_n_br1_showers", "shw_sp_n_br2_showers", 
                  "shw_sp_n_br3_showers", "shw_sp_n_br4_showers", "shw_sp_n_20br1_showers", 
                  "shw_sp_shw_vtx_dis", "shw_sp_max_shw_dis"]

#other_bkg_vars += taggerHOL_var
#other_bkg_vars += taggerBR1_var
#other_bkg_vars += taggerBR3_var
other_bkg_vars += taggerBR2_var
other_bkg_vars += taggerBR4_var
other_bkg_vars += taggerLEM_var
other_bkg_vars += taggerMIP_var
other_bkg_vars += taggerMIPdqdx_var
other_bkg_vars += ["shw_sp_proton_length_1", "shw_sp_proton_dqdx_1", "shw_sp_proton_energy_1",
                    "shw_sp_proton_length_2", "shw_sp_proton_dqdx_2", "shw_sp_proton_energy_2",
                    "shw_sp_n_good_showers", "shw_sp_n_br1_showers", "shw_sp_n_br2_showers", 
                   "shw_sp_n_br3_showers", "shw_sp_n_br4_showers", "shw_sp_n_20br1_showers", 
                    "shw_sp_shw_vtx_dis", "shw_sp_max_shw_dis"]
other_bkg_vars += ["shw_sp_num_mip_tracks","shw_sp_num_muons","shw_sp_num_protons"]


#remaining_vars += ["shw_sp_proton_length_1", "shw_sp_proton_dqdx_1", "shw_sp_proton_energy_1",
#                      "shw_sp_proton_length_2", "shw_sp_proton_dqdx_2", "shw_sp_proton_energy_2",
#                      "shw_sp_n_good_showers", "shw_sp_n_br1_showers", "shw_sp_n_br2_showers", 
#                     "shw_sp_n_br3_showers", "shw_sp_n_br4_showers", "shw_sp_n_20br1_showers", 
#                       "shw_sp_shw_vtx_dis", "shw_sp_max_shw_dis"]
#remaining_vars += ["shw_sp_num_mip_tracks","shw_sp_num_muons","shw_sp_num_protons"]

#ncpi0_bkg_vars += taggerHOL_var
#ncpi0_bkg_vars += taggerBR2_var
#ncpi0_bkg_vars += taggerBR4_var
#ncpi0_bkg_vars += taggerPI01_var
#ncpi0_bkg_vars += taggerAdditional_var
#ncpi0_bkg_vars += taggerNumTrk_var

#ncpi0_bkg_vars += ["shw_sp_pio_flag_pio"]
#ncpi0_bkg_vars += ["shw_sp_n_20br1_showers_calc"]
#ncpi0_bkg_vars += ["shw_sp_E_indirect_max_energy","shw_sp_lol_3_vtx_n_segs"]
ncpi0_bkg_vars += taggerPI01_var#["shw_sp_pio_flag_pio"]
ncpi0_bkg_vars += ["shw_sp_n_20br1_showers","shw_sp_n_br1_showers", "shw_sp_n_br2_showers", 
                   "shw_sp_n_br3_showers", "shw_sp_n_br4_showers"]
ncpi0_bkg_vars += ["shw_sp_E_indirect_max_energy","shw_sp_lol_3_vtx_n_segs", "shw_sp_hol_1_min_length"]

nue_bkg_vars += taggerMIP_var
nue_bkg_vars += taggerMIPdqdx_var
nue_bkg_vars += taggerAdditional_var

#remaining_vars += taggerAdditional_var

bdt_variables+=all_sp_scalars

extratop_var = ["shw_sp_hol_1_min_angle","shw_sp_hol_1_energy",
                 "shw_sp_hol_2_min_angle",
                 "shw_sp_lol_3_angle_beam","shw_sp_lol_3_min_angle",
                "shw_sp_br1_1_energy","shw_sp_br1_1_flag_sg_topology", "shw_sp_br1_1_flag_sg_trajectory",
                "shw_sp_br3_1_energy", "shw_sp_br3_1_sg_flag_trajectory",
                 "shw_sp_br3_1_iso_angle", "shw_sp_br3_1_sg_flag_topology",
                "shw_sp_br4_1_energy","shw_sp_br4_2_iso_angle",
                 "shw_sp_br4_2_iso_angle1","shw_sp_br4_2_angle",
                "shw_sp_br2_max_angle","shw_sp_br2_flag_sg_trajectory",
                "shw_sp_iso_angle","shw_sp_flag_stem_trajectory"]

other_bkg_vars += ["shw_sp_hol_2_min_angle"]#, "shw_sp_hol_1_min_angle"]
ncpi0_bkg_vars += ["shw_sp_hol_2_min_angle", "shw_sp_hol_1_min_angle"]

#remaining_vars += ["shw_sp_hol_1_min_angle"]

#bdt_variables+=extratop_var
#bdt_variables += ["shw_sp_hol_2_min_angle", "shw_sp_hol_1_min_angle"]

numu_bdt_score_variables = [
    "cosmict_10_score",
    "numu_1_score",
    "numu_2_score",
    "numu_score"
]

nue_bdt_score_variables = [
    "tro_5_score",
    "tro_4_score",
    "tro_2_score",
    "tro_1_score",
    "stw_4_score",
    "stw_3_score",
    "stw_2_score",
    "sig_2_score",
    "sig_1_score",
    "pio_2_score",
    "lol_2_score",
    "lol_1_score",
    "br3_6_score",
    "br3_5_score",
    "br3_3_score"
]

numu_bkg_vars += ["numu_1_score", "numu_score"]

other_bkg_vars += numu_bdt_score_variables
other_bkg_vars += ["reco_nuvtxY"]

#remaining_vars += nue_bdt_score_variables

ncpi0_bkg_vars += nue_bdt_score_variables

#bdt_variables+=numu_bdt_score_variables
#bdt_variables+=nue_bdt_score_variables

single_photon_score_variables = [
    "single_photon_numu_score",
    "single_photon_other_score",
    "single_photon_ncpi0_score",
    "single_photon_nue_score"
]

bdt_variables+=single_photon_score_variables
bdt_variables+=["shw_sp_energy","shw_sp_angle_beam","shw_sp_flag"]

scalar_kine_variables = [
    #"kine_reco_Enu",
    #"kine_reco_add_energy",
    "kine_pio_mass",
    "kine_pio_flag",
    "kine_pio_vtx_dis",
    "kine_pio_energy_1",
    "kine_pio_theta_1",
    "kine_pio_phi_1",
    "kine_pio_dis_1",
    "kine_pio_energy_2",
    "kine_pio_theta_2",
    "kine_pio_phi_2",
    "kine_pio_dis_2",
    "kine_pio_angle"
]

other_kine_variables = [
    "kine_reco_Enu",
    "kine_reco_add_energy",
    "kine_energy_particle",
    "kine_particle_type"
]

ncpi0_bkg_vars += scalar_kine_variables

kine_variables = []
kine_variables += scalar_kine_variables
kine_variables += other_kine_variables

pfeval_variables = [
    "run",
    "subrun",
    "event",
    #"true_event_type",
    #"weight",
    #"numu_score"
    #"numu_cc_1_particle_type",
    #"num_showers",
   # "truth_pio_energy_1",
   # "truth_pio_energy_2",
    "reco_nuvtxX",
    "reco_nuvtxY",
    "reco_nuvtxZ",
    "reco_showervtxX",
    "reco_showervtxY",
    "reco_showervtxZ",
    "reco_showerMomentum",
    "reco_muonMomentum"
]

truth_variables = [
    "nuvtx_diff",
    "showervtx_diff",
    "muonvtx_diff",
    "truth_isCC",
    #"truth_vtxInside",
    #"truth_nuPdg",
    #"truth_nuEnergy",
    "truth_nuIntType",
    #"truth_energyInside",
    #"weight_spline",
    #"weight_cv",
    #"event_type",
    #"lowEweight",
    "truth_single_photon",
    "truth_muonMomentum",
    "truth_showerMother",
    "truth_Npi0",
    "truth_NCDelta",
    "truth_showerKE",
    "truth_showerMomentum"
]

eval_variables = [
    "match_isFC",
    "match_energy"
]
eval_truth_variables = [
    "match_completeness_energy",
    "truth_energyInside",
    "truth_nuPdg",
    #"truth_isCC",
    "truth_vtxInside",
    "stm_eventtype",
    "truth_nuEnergy",
    "weight_cv",
    "weight_spline",
    "weight_lee"
]

time_variables = ["evtTimeNS","evtDeltaTimeNS"]