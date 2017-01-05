#test levenshtein dist before and after sequencing
#here using whole songs to create transition matrix
import sys, os, ast # ast for literal_eval, to parse .txt files as lists

import pandas as pd
import numpy as np

from hvc.utils import parse_xml, lev_np, resequencer, frame_error, get_trans_mat

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

columns=['sum_dists_val_seqs',
        'sum_dists_val_songs']
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
    train_seqs = parse_xml('TrainingSequenceBdLcGs.xml')

    seq_list = parse_xml('Annotation.xml')
    all_syls = [syl.label for seq in seq_list for syl in seq.syls]
    uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)

    trans_mat = get_trans_mat(train_seqs)
    reseqr_seqs = resequencer(trans_mat,uniq_syls)

    train_songs = parse_xml('TrainingSequenceBdLcGs.xml',
                            concat_seqs_into_songs=True)
    song_trans_mat = get_trans_mat(train_songs)
    reseqr_song = resequencer(song_trans_mat,uniq_syls)

    obsv_prob = []
    with open('observation_prob.txt') as op_file:
        lines = op_file.readlines()

    for line in lines:
        probs_arr = np.asarray(ast.literal_eval(line))
        obsv_prob.append(probs_arr)

    seqs_after_reseq = []
    for ind,obsv_seq in enumerate(obsv_prob):
        sys.stdout.write(
        '\rresequencing {} of {} with seq trans mat'.format(ind,len(obsv_prob)))
        sys.stdout.flush()
        seqs_after_reseq.append(reseqr_seqs.resequence(obsv_seq))

    val_syls = get_syl_list(val_seqs)
    dists_val_seqs = get_dist_list(val_syls,seqs_after_reseq)
    results_dict['sum_dists_val_seqs'].append(sum(dists_val_seqs))
    print('. Distance from validation set: {}'.format(
                        results_dict['sum_dists_val_seqs'][-1]))

    seqs_after_reseq_song = []
    for ind,obsv_seq in enumerate(obsv_prob):
        sys.stdout.write(
        '\rresequencing {} of {} with song trans mat'.format(ind,len(obsv_prob)))
        sys.stdout.flush()
        seqs_after_reseq_song.append(reseqr_song.resequence(obsv_seq))

    dists_val_seq_songs = get_dist_list(val_syls,seqs_after_reseq_song)
    results_dict['sum_dists_val_songs'].append(sum(dists_val_seq_songs))
    print('. Distance from validation set: {}'.format(
                        results_dict['sum_dists_val_songs'][-1]))

results_df = pd.DataFrame(results_dict,index=bird_names)
results_df.to_csv(ROOT_DIR_NAME + 'resequence_with_song_trans_mat_results.csv')
