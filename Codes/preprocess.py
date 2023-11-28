# -------------------------------------------------------------------------
# Name: preprocess.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Extracting information (TCRb CDR3 sequence, v_gene, frequency)
#        from raw TCR sequencing files, and predicting each TCR.
#        The processed file's format is that:
#        ----- TCR_seq.tsv_processed.tsv -----
#        amino_acid v_gene  frequency   target_seq   [caTCR_score]
#        CASRGRGWDTEAFF TRBV19*01 0.010528  CASRGRGWDTEAFF  [0.2238745]
#        ......
# -------------------------------------------------------------------------

import argparse
import os

import utils
from network import TCRD


def create_parser():
    parser = argparse.ArgumentParser(
        description='Script to preprocess raw TCR sequencing files.'
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of samples for preprocessing.",
        required=True
    )
    parser.add_argument(
        "--info_index",
        dest="info_index",
        type=str,
        help="The index of information (aaSeqCDR3, allVHitsWithScore, cloneFraction) in the raw files.",
        required=True
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs extracted from the raw files.",
        default=-1,
    )
    parser.add_argument(
        "--crop_num",
        dest="crop_num",
        type=int,
        help="The number of amino acids discarded at the beginning/end of the TCR sequences."
             "E.g., if crop_num = 2, 'CASSFIRLGDSGYTF' => 'SSFIRLGDSGY'.",
        default=0,
    )
    parser.add_argument(
        "--filters_num",
        dest="filters_num",
        type=int,
        help="The number of the filter set in DeepLION.",
        default=1,
    )
    parser.add_argument(
        "--aa_file",
        dest="aa_file",
        type=str,
        help="The file recording animo acid vectors.",
        required=True,
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pre-trained TCRD model file of for TCR prediction in .pth format. "
             "The default value 'None' means no prediction for TCRs.",
        default=None
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        type=str,
        help="The directory to save preprocessed files.",
        required=True
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cpu",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        type=str,
        help="The ratio to split data.",
        default="[1]"
    )
    args = parser.parse_args()
    return args


def preprocess_files(sp_dir, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio):
    sv_dir = sv_dir + utils.get_last_path(sp_dir)
    utils.create_directory(sv_dir)
    for e in os.listdir(sp_dir):
        e_path = utils.correct_path(sp_dir + e)
        if os.path.isdir(e_path):
            preprocess_files(e_path, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio)
        else:
            preprocess_file(e_path, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir)
    utils.split_data(sv_dir, ratio)
    return 0


def preprocess_file(fname, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir):
    # Extract information.
    extract_info = utils.read_tsv(fname, info_i, True)
    # Filter out invalid TCR sequences.
    filtered_info = []
    for tcr in extract_info:
        # Check TCR.
        if not utils.check_tcr(tcr[0]):
            continue
        # Check V gene.
        if not tcr[1]:
            continue
        else:
            tcr[1] = process_vgene(tcr[1])
        # Check frequency.
        if not tcr[2]:
            continue
        if crop_n != 0:
            filtered_info.append([tcr[0], tcr[1], tcr[2], tcr[0][crop_n: -crop_n]])
        else:
            filtered_info.append([tcr[0], tcr[1], tcr[2], tcr[0]])
    # Sort TCRs by freqency and get top k TCRs.
    filtered_info = sorted(filtered_info, key=lambda x: float(x[2]), reverse=True)
    if 0 < tcr_n < len(filtered_info):
        filtered_info = filtered_info[: tcr_n]
    # Prediction for TCRs
    tcr_scores = []
    if model_f:
        # Get cropped TCR sequences.
        input_tcrs = []
        for tcr in filtered_info:
            input_tcrs.append(tcr[3])
        tcr_scores = TCRD.prediction(input_tcrs, filters_n, model_f, aa_f, device)
    # Save results.
    sv_fname = sv_dir + utils.get_last_path(fname)
    if model_f:
        sv_fname += "_processed_TCRD.tsv"
    else:
        sv_fname += "_processed.tsv"
    with open(sv_fname, "w", encoding="utf8") as wf:
        if not model_f:
            wf.write("amino_acid\tv_gene\tfrequency\ttarget_seq\n")
        else:
            wf.write("amino_acid\tv_gene\tfrequency\ttarget_seq\tcaTCR_score\n")
        for ind, tcr in enumerate(filtered_info):
            wf.write("\t".join(tcr))
            if tcr_scores:
                wf.write("\t{0}".format(tcr_scores[ind]))
            wf.write("\n")
    return 0


def process_vgene(vgene):
    # Process multiple v gene with scores.
    vgene_list = vgene.split(",")
    if len(vgene_list) > 1:
        final_gene, max_score = "", 0
        for vg in vgene_list:
            vg = vg.strip()[: -1]
            v, s = vg.split("(")
            if float(s) > max_score:
                max_score = float(s)
                final_gene = v.strip()
    else:
        final_gene = vgene_list[0]
    return final_gene


def main():
    # Parse arguments.
    args = create_parser()
    args.sample_dir = utils.correct_path(args.sample_dir)
    args.aa_file = utils.correct_path(args.aa_file)
    if args.model_file:
        args.model_file = utils.correct_path(args.model_file)
    args.save_dir = utils.correct_path(args.save_dir)
    while type(args.info_index) == str:
        args.info_index = eval(args.info_index)
    if type(args.info_index) != list:
        print("Wrong parameter 'info_index' set!")
        return -1
    while type(args.ratio) == str:
        args.ratio = eval(args.ratio)
    if type(args.ratio) != list:
        print("Wrong parameter 'ratio' set!")
        return -1

    # Preprocess raw fils.
    utils.create_directory(args.save_dir)
    preprocess_files(args.sample_dir, args.info_index, args.tcr_num, args.crop_num, args.filters_num, args.aa_file,
                     args.model_file, args.device, args.save_dir, args.ratio)


if __name__ == "__main__":
    main()
