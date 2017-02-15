#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import random
import itertools

def read_images_from_folder(input_folder):
    filenames = []
    for filename in os.listdir(input_folder):
        filename = os.path.join(input_folder, filename)
        if os.path.isdir(filename):
            print filename
            continue
        
        if filename[-4:] != '.jpg' and filename[-4:] != '.png':
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

def generate_triplet_data(input_images, pair_size, valid_ratio=0.10):
    train_triplets, valid_triplets = [], []
    for idx, images in enumerate(input_images):
        pair_data_list = list(itertools.combinations(images, pair_size))
        print len(images), 'images and', len(pair_data_list), 'pairs in', idx
        random.shuffle(pair_data_list)
        valid_cnt = int(len(pair_data_list) * valid_ratio)

        for pair_idx, pair_data in enumerate(pair_data_list):
            triplets = ['%s %d' % (filename, idx) for filename in pair_data]
            assert len(triplets) == pair_size
            if pair_idx < valid_cnt:
                valid_triplets.extend(triplets)
            else:
                train_triplets.extend(triplets)
    return train_triplets, valid_triplets


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python gen_triplet_data.py [folder] [pair_size]'
        sys.exit(-1)
    
    face_path = sys.argv[1]
    pair_size = int(sys.argv[2])

    X = read_images(face_path)
    train_triplets, valid_triplets = generate_triplet_data(X, pair_size)
    with open('./train_triplets.txt', 'w') as f:
        f.writelines('\n'.join(train_triplets))
    with open('./valid_triplets.txt', 'w') as f:
        f.writelines('\n'.join(valid_triplets))
