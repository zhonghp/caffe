#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: vasezhong
# @Date:   2017-02-22 15:09:55
# @Last Modified by:   vasezhong
# @Last Modified time: 2017-02-22 15:37:00

import os
import sys
import random

def read_images_from_folder(input_folder):
    filenames = []
    for filename in os.listdir(input_folder):
        filename = os.path.join(input_folder, filename)
        if os.path.isdir(filename):
            print filename
            continue

        _, ext = os.path.splitext(filename)
        if ext != '.jpg' and ext != '.jpeg' and ext != '.png':
            print filename
            continue
        filenames.append(os.path.abspath(filename))
    return filenames

def read_images(input_folder):
    X = []
    for folder in os.listdir(input_folder):
        image_folder = os.path.join(input_folder, folder)
        filenames = read_images_from_folder(image_folder)
        X.append(filenames)
    lens = map(len, X)
    print 'Totally', len(X), 'folders.'
    print min(lens), max(lens)
    return X

def generate_pair_data(input_images, valid_cnt=10000):
    pairs = []
    for idx, images in enumerate(input_images):
        for image in images:
            pos = image
            while pos == image:
                pos = random.choice(images)

            neg_idx = idx
            while neg_idx == idx:
                neg_idx = random.randint(0, len(input_images)-1)
            neg = random.choice(input_images[neg_idx])

            pairs.append((image, idx, pos, idx))
            pairs.append((image, idx, neg, neg_idx))
    print 'Totally', len(pairs), 'pairs'
    random.shuffle(pairs)

    train_pairs, train_pairs_anc = [], []
    valid_pairs, valid_pairs_anc = [], []
    for idx, pair in enumerate(pairs):
        assert len(pair) == 4
        temp_pair = '%s %d' % (pair[0], pair[1])
        temp_pair_anc = '%s %d' % (pair[2], pair[3])
        if idx < valid_cnt:
            valid_pairs.append(temp_pair)
            valid_pairs_anc.append(temp_pair_anc)
        else:
            train_pairs.append(temp_pair)
            train_pairs_anc.append(temp_pair_anc)

    return train_pairs, train_pairs_anc, valid_pairs, valid_pairs_anc


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python gen_pair_data.py [folder]'
        sys.exit(-1)

    face_path = sys.argv[1]

    X = read_images(face_path)
    train_pairs, train_pairs_anc, valid_pairs, valid_pairs_anc = generate_pair_data(X)
    with open('./ident_verif_train.txt', 'w') as f:
        f.writelines('\n'.join(train_pairs))
    with open('./ident_verif_train_p.txt', 'w') as f:
        f.writelines('\n'.join(train_pairs_anc))
    with open('./ident_verif_test.txt', 'w') as f:
        f.writelines('\n'.join(valid_pairs))
    with open('./ident_verif_test_p.txt', 'w') as f:
        f.writelines('\n'.join(valid_pairs_anc))
