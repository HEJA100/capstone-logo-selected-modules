import os, sys
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

"""
00_DataPrepare.py

@author: liwenran
"""

###################### Input #######################
if len(sys.argv) < 3:
    print('[USAGE] python 00_DataPrepare.py cell interaction_type')  # py2改为py3
    print('For example, python 00_DataPrepare.py Mon P-E')
    sys.exit()
# 修改后：
CELL = sys.argv[1]
TYPE = sys.argv[2]
if len(sys.argv) >= 4:
    TASK = sys.argv[3]
else:
    TASK = 'train' # 如果没给，默认做 train

if TYPE == 'P-P':
    RESIZED_LEN = 1000  # promoter
elif TYPE == 'P-E':
    RESIZED_LEN = 2000  # enhancer
else:
    print('[USAGE] python 00_DataPrepare.py cell interaction_type')
    print('For example, python 00_DataPrepare.py Mon P-E')
    sys.exit()

chr_dict = {"NC_000001.10": "chr1",
            "NC_000002.11": "chr2",
            "NC_000003.11": "chr3",
            "NC_000004.11": "chr4",
            "NC_000005.9": "chr5",
            "NC_000006.11": "chr6",
            "NC_000007.13": "chr7",
            "NC_000008.10": "chr8",
            "NC_000009.11": "chr9",
            "NC_000010.10": "chr10",
            "NC_000011.9": "chr11",
            "NC_000012.11": "chr12",
            "NC_000013.10": "chr13",
            "NC_000014.8": "chr14",
            "NC_000015.9": "chr15",
            "NC_000016.9": "chr16",
            "NC_000017.10": "chr17",
            "NC_000018.9": "chr18",
            "NC_000019.9": "chr19",
            "NC_000020.10": "chr20",
            "NC_000021.8": "chr21",
            "NC_000022.10": "chr22",
            "NC_000023.10": "chrX",
            "NC_000024.9": "chrY"}


def split():
    pairs = pd.read_csv(CELL + '/' + TYPE + '/pairs.csv')
    n_sample = pairs.shape[0]
    rand_index = list(range(0, n_sample))
    np.random.seed(n_sample)
    np.random.shuffle(rand_index)

    n_sample_train = n_sample - n_sample // 10  # Take 90% as train
    pairs_train = pairs.iloc[rand_index[:n_sample_train]]  # Divide the data set
    pairs_test = pairs.iloc[rand_index[n_sample_train:]]

    # imbalanced testing set
    pairs_test_pos = pairs_test[pairs_test['label'] == 1]  # Take pos and nneg in tets
    pairs_test_neg = pairs_test[pairs_test['label'] == 0]
    num_pos = pairs_test_pos.shape[0]
    num_neg = pairs_test_neg.shape[0]

    np.random.seed(num_neg)
    rand_index = list(range(0, num_neg))
    pairs_test_neg = pairs_test_neg.iloc[rand_index[:num_pos * 5]]
    pairs_test = pd.concat([pairs_test_pos, pairs_test_neg])

    # save
    pairs_train.to_csv(CELL + '/' + TYPE + '/pairs_train.csv', index=False)
    print("Writting ", CELL + '/' + TYPE + '/pairs_train.csv')
    pairs_test.to_csv(CELL + '/' + TYPE + '/pairs_test.csv', index=False)
    print("Writting ", CELL + '/' + TYPE + '/pairs_test.csv')


def resize_location(original_location, resize_len):
    original_len = int(original_location[1]) - int(original_location[0])
    len_diff = abs(resize_len - original_len)
    rand_int = np.random.randint(0, len_diff + 1)
    if resize_len < original_len: rand_int = - rand_int
    resize_start = int(original_location[0]) - rand_int
    resize_end = resize_start + resize_len
    return (str(resize_start), str(resize_end))


def augment():
    RESAMPLE_TIME = 20
    PROMOTER_LEN = 1000
    in_path = CELL + '/' + TYPE + '/pairs_train.csv'
    out_path = CELL + '/' + TYPE + '/pairs_train_augment.csv'

    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        header = fin.readline().strip()
        fout.write(header + '\n')

        for raw in fin:
            line = raw.strip().split(',')
            if len(line) < 9:
                continue

            if line[-1] != '1':
                fout.write(','.join(line[:9]) + '\n')
                continue

            for _ in range(0, RESAMPLE_TIME):
                enh_start, enh_end = resize_location((line[1], line[2]), RESIZED_LEN)
                pro_start, pro_end = resize_location((line[5], line[6]), PROMOTER_LEN)

                out_fields = [
                    line[0], enh_start, enh_end, line[3],
                    line[4], pro_start, pro_end, line[7],
                    line[8],
                ]
                fout.write(','.join(out_fields) + '\n')

    print("Finished:", out_path)


def one_hot(sequence_dict, chrom, start, end, chr_convert_dict: dict = {}):
    seq_dict = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
                'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                'a': [1, 0, 0, 0], 'g': [0, 1, 0, 0],
                'c': [0, 0, 1, 0], 't': [0, 0, 0, 1]}
    temp = []

    seq = str(sequence_dict[chr_convert_dict[chrom]].seq[start:end])
    seq = seq.upper()
    # print("seq")
    for c in seq:
        temp.extend(seq_dict.get(c, [0, 0, 0, 0]))
    temp = np.array(temp)
    # print("temp: ", temp.shape)
    return temp


def encoding(sequence_dict, filename, PROMOTER_LEN=1000, NUM_SEQ=4, task: str = None):
    in_path = CELL + '/' + TYPE + '/' + filename
    retained_path = CELL + '/' + TYPE + '/' + filename.replace('.csv', '.retained.csv')

    file = open(in_path)
    header = file.readline().strip()

    retained_fout = open(retained_path, 'w')
    retained_fout.write(header + '\n')

    seqs_1 = []
    seqs_2 = []
    label = []

    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[v] = k

    # Extract sequence one by one
    ii = 0
    for raw in tqdm(file):
        raw = raw.strip()
        if len(raw) == 0:
            continue
        line = raw.split(',')
        if len(line) == 10 and line[-1] == '':
            line = line[:-1]
        if len(line) < 9:
            continue

        seq_1 = one_hot(sequence_dict, line[0], int(line[1]), int(line[2]), chr_convert_dict)
        seq_2 = one_hot(sequence_dict, line[4], int(line[5]), int(line[6]), chr_convert_dict)

        if len(seq_1) != RESIZED_LEN * NUM_SEQ or len(seq_2) != PROMOTER_LEN * NUM_SEQ:
            print(len(seq_1), len(seq_2))
            continue

        if len(line[-1]) == 0:
            if len(line[-2]) > 0:
                label.append(int(line[-2]))  # The last one is label
            else:
                print(line)
                continue
        else:
            label.append(int(line[-1]))  # The last one is label

        seqs_1.append(seq_1)  # Extract the first sequence (such as P)
        seqs_2.append(seq_2)  # Extract the second sequence (such as E)
        retained_fout.write(','.join(line[:9]) + '\n')

        ii += 1

        if len(seqs_1) % 50000 == 0:
            if TYPE == 'P-P':
                print("promoter1_Seq, promoter2_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/promoter1_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_1))
                np.savez(CELL + '/' + TYPE + '/promoter2_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_2))
            else:
                print("enhancer_Seq, promoter_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/enhancer_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_1))
                np.savez(CELL + '/' + TYPE + '/promoter_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_2))

            seqs_1 = []
            seqs_2 = []
            label = []

    if TYPE == 'P-P':
        print("promoter1_Seq, promoter2_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/promoter1_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_1))
        np.savez(CELL + '/' + TYPE + '/promoter2_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_2))
    else:
        print("enhancer_Seq, promoter_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/enhancer_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_1))
        np.savez(CELL + '/' + TYPE + '/promoter_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_2))
    retained_fout.close()


def encoding_test(sequence_dict, filename, PROMOTER_LEN=1000, NUM_SEQ=4, task: str = None):
    in_path = CELL + '/' + TYPE + '/' + filename

    if os.path.exists(CELL + '/' + TYPE + '/test/') is False:
        os.makedirs(CELL + '/' + TYPE + '/test/')

    retained_path = CELL + '/' + TYPE + '/test/' + filename.replace('.csv', '.retained.csv')

    file = open(in_path)
    header = file.readline().strip()

    retained_fout = open(retained_path, 'w')
    retained_fout.write(header + '\n')

    seqs_1 = []
    seqs_2 = []
    label = []

    chr_convert_dict = {}
    for k, v in chr_dict.items():
        chr_convert_dict[v] = k

    if os.path.exists(CELL + '/' + TYPE + '/test/') is False:
        os.makedirs(CELL + '/' + TYPE + '/test/')

    # Extract sequence one by one
    ii = 0
    for raw in tqdm(file):
        raw = raw.strip()
        if len(raw) == 0:
            continue
        line = raw.split(',')
        if len(line) == 10 and line[-1] == '':
            line = line[:-1]
        if len(line) < 9:
            continue

        seq_1 = one_hot(sequence_dict, line[0], int(line[1]), int(line[2]), chr_convert_dict)
        seq_2 = one_hot(sequence_dict, line[4], int(line[5]), int(line[6]), chr_convert_dict)

        if len(seq_1) != RESIZED_LEN * NUM_SEQ or len(seq_2) != PROMOTER_LEN * NUM_SEQ:
            print(len(seq_1), len(seq_2))
            continue

        if len(line[-1]) == 0:
            if len(line[-2]) > 0:
                label.append(int(line[-2]))  # The last one is label
            else:
                print(line)
                continue
        else:
            label.append(int(line[-1]))  # The last one is label

        seqs_1.append(seq_1)  # Extract the first sequence (such as P)
        seqs_2.append(seq_2)  # Extract the second sequence (such as E)
        retained_fout.write(','.join(line[:9]) + '\n')

        ii += 1

        if len(seqs_1) % 50000 == 0:
            if TYPE == 'P-P':
                print("promoter1_Seq, promoter2_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/test/promoter1_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_1))
                np.savez(CELL + '/' + TYPE + '/test/promoter2_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_2))
            else:
                print("enhancer_Seq, promoter_Seq, label shape : ",
                      np.array(seqs_1).shape,
                      np.array(seqs_2).shape,
                      np.array(label).shape)
                np.savez(CELL + '/' + TYPE + '/test/enhancer_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_1))
                np.savez(CELL + '/' + TYPE + '/test/promoter_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                         sequence=np.array(seqs_2))

            seqs_1 = []
            seqs_2 = []
            label = []

    if TYPE == 'P-P':
        print("promoter1_Seq, promoter2_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/test/promoter1_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_1))
        np.savez(CELL + '/' + TYPE + '/test/promoter2_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_2))
    else:
        print("enhancer_Seq, promoter_Seq, label shape : ",
              np.array(seqs_1).shape,
              np.array(seqs_2).shape,
              np.array(label).shape)
        np.savez(CELL + '/' + TYPE + '/test/enhancer_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_1))
        np.savez(CELL + '/' + TYPE + '/test/promoter_Seq_{}.npz'.format(str(ii)), label=np.array(label),
                 sequence=np.array(seqs_2))
    retained_fout.close()


def main():
    if TASK == 'train':
        """Split for training and testing data"""
        print("Split for training and testing data")
        split()
        """Augment training data"""
        print("Augment training data")
        augment()
        """One-hot encoding"""
        print("One-hot encoding")

        reffasta = '../data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(reffasta), 'fasta'))
        # sequence_dict = SeqIO.to_dict(SeqIO.parse(open('E:/myP/ExPecto/resources/hg19.fa'), 'fasta'))
        encoding(sequence_dict, 'pairs_train_augment.csv')
        print("Finished!")
    else:
        print("One-hot encoding")
        reffasta = '../data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna'
        sequence_dict = SeqIO.to_dict(SeqIO.parse(open(reffasta), 'fasta'))
        # sequence_dict = SeqIO.to_dict(SeqIO.parse(open('E:/myP/ExPecto/resources/hg19.fa'), 'fasta'))
        encoding_test(sequence_dict, 'pairs_test.csv')
        print("Finished!")


"""RUN"""
main()
