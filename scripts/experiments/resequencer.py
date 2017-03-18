#test levenshtein dist before and after sequencing
import sys, os, ast # ast for literal_eval, to parse .txt files as lists

import pandas as pd
import numpy as np

from hvc.utils import parse_xml, lev_np, resequencer, frame_error

def get_syl_list(seq_list):
    """
    create list of strings representing labels from sequences
    """
    syl_list = []
    for seq in seq_list:
        syl_list.append([syl.label for syl in seq.syls])
    return syl_list

def get_dist_list(source_list,target_list):
    """
    get list of Levenshtein distances
    """
    dist_list = []
    for source_string, target_string in zip(source_list,target_list):
        dist_list.append(lev_np(source_string,target_string))
    return dist_list

# main script
ROOT_DIR_NAME = 'c:/DATA/koumura birds/'
dir_names = {}
bird_IDs = [str(num) for num in range(0,11)]
bird_names = []
for bird_id in bird_IDs:
    bird_name = 'Bird' + bird_id
    bird_names.append(bird_name)
    tmp_dir = ROOT_DIR_NAME + bird_name
    dir_names[bird_name] = tmp_dir

columns=['sum_dists_b4',
        'sum_dists_aft',
        'sum_dists_java_py',
        'sum_dists_val_py',
        'b4_frame_err',
        'aft_frame_err']
results_dict = {}
for column in columns:
    results_dict[column] = []

for bird_name in bird_names:

    os.chdir(dir_names[bird_name])
    print("processing {}".format(dir_names[bird_name]))

    # first test output from Koumura code
    out_b4_hmm_seqs = parse_xml('OutputSequenceBeforeHmm.xml')
    out_aft_hmm_seqs = parse_xml('OutputSequenceBdLcGs.xml')
    val_seqs = parse_xml('ValidationSequenceBdLcGs.xml')

    out_b4_syls = get_syl_list(out_b4_hmm_seqs)
    out_aft_syls = get_syl_list(out_aft_hmm_seqs)
    val_syls = get_syl_list(val_seqs)

    dists_b4 = get_dist_list(val_syls,out_b4_syls)
    dists_aft = get_dist_list(val_syls,out_aft_syls)

    results_dict['sum_dists_b4'].append(sum(dists_b4))
    results_dict['sum_dists_aft'].append(sum(dists_aft))

    seq_list = parse_xml('Annotation.xml')
    all_syls = [syl.label for seq in seq_list for syl in seq.syls]
    uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)

    with open('transition_prob.txt') as tp_file:
        line = tp_file.readline()  # file is only one line
    trans_prob = np.asarray(ast.literal_eval(line))

    reseqr = resequencer(trans_prob,uniq_syls)

    obsv_prob = []
    with open('observation_prob.txt') as op_file:
        lines = op_file.readlines()

    for line in lines:
        probs_arr = np.asarray(ast.literal_eval(line))
        obsv_prob.append(probs_arr)

    seqs_after_reseq = []
    for ind,obsv_seq in enumerate(obsv_prob):
        sys.stdout.write(
        '\rresequencing {} of {}'.format(ind,len(obsv_prob)))
        sys.stdout.flush()
        seqs_after_reseq.append(reseqr.resequence(obsv_seq))

    dists_java_py = get_dist_list(out_aft_syls,seqs_after_reseq)
    results_dict['sum_dists_java_py'].append(sum(dists_java_py))

    dists_val_py = get_dist_list(val_syls,seqs_after_reseq)
    results_dict['sum_dists_val_py'].append(sum(dists_val_py))
    print("")
    print("Sum of Levenshtein distances between validation set and output of DCNN"
          " before resequencing with HMM: {}".format(
            results_dict['sum_dists_b4'][-1]))
    print("Sum of Levenshtein distances between validation set and output of DCNN"
          " after resequencing with HMM: {}".format(
            results_dict['sum_dists_aft'][-1]))
    print("Sum of Levenshtein distances between"
          " output of DCNN after resequencing with HMM and "
          "after resequencing with Python implementation: {}".format(
            results_dict['sum_dists_java_py'][-1]))
    print("Sum of Levenshtein distances between validation set and"
          " output of DCNN after resequencing after resequencing with Python"
          " implementation: {}".format(
            results_dict['sum_dists_val_py'][-1]))
    val_labels = [seq_syls for seq in val_syls for seq_syls in seq]
    out_b4_labels = [seq_syls for seq in out_b4_syls for seq_syls in seq]

    results_dict['b4_frame_err'].append(frame_error(val_seqs,out_b4_hmm_seqs))
    results_dict['aft_frame_err'].append(frame_error(val_seqs,out_aft_hmm_seqs))
    print("b4 err: {}".format(results_dict['b4_frame_err'][-1]))
    print("aft4 err: {}".format(results_dict['aft_frame_err'][-1]))

results_df = pd.DataFrame(results_dict,index=bird_names)
results_df['frame_error_diff'] = results_df['b4_frame_err'] - results_df['aft_frame_err']
results_df.to_csv(ROOT_DIR_NAME + 'resequencer_results.csv')
