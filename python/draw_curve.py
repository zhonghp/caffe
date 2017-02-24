#!/usr/bin/env python
# encoding: utf-8

"""
@file: draw_curve.py
@version: 0.1.0

@author: zhonghp
@time: 7/26/16 11:39 AM
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print 'Usage: python draw_curve.py [log_file] [display] [test_interval]'
        print 'Usage: python draw_curve.py [log_file] [display] [test_interval] [test_iter]'
        return

    log_file = sys.argv[1]
    display = int(sys.argv[2])
    test_interval = int(sys.argv[3])
    # print log_file, display, test_interval

    # train_losses_shell = "cat {} | grep ' Iteration [0-9]*, loss = ' | awk '{}'".format(log_file, '{print $9}')
    train_losses_shell = "cat {} | grep ' Iteration [0-9]* ' | awk '{}'".format(log_file, '{print $13}')
    train_losses = os.popen(train_losses_shell).readlines()
    train_losses = map(eval, train_losses)
    idx_train_losses = [idx*display for idx in xrange(len(train_losses))]

    print idx_train_losses
    print train_losses

    test_losses_shell = "cat {} | grep 'Test loss: ' | awk '{}'".format(log_file, '{print $7}')
    test_losses = os.popen(test_losses_shell).readlines()
    test_losses = map(eval, test_losses)
    idx_test_losses = [idx*test_interval for idx in xrange(len(test_losses))]

    print idx_test_losses
    print test_losses

    if len(sys.argv) == 5:
        test_iter = int(sys.argv[4])
        triplet_cnt_shell = "cat {} | grep ' Totally ' | awk '{}'".format(log_file, '{print $6}')
        triplet_cnts = os.popen(triplet_cnt_shell).readlines()
        triplet_cnts = map(eval, triplet_cnts)

        # print triplet_cnts
        cnt = test_iter + test_interval
        train_cnt = test_interval // display
        train_test_cnt = len(triplet_cnts) // cnt

        print len(triplet_cnts), cnt, train_test_cnt
        test_triplet_cnts = []
        train_triplet_cnts = []
        for i in range(train_test_cnt):
            test_beg_idx = i*cnt
            test_end_idx = test_beg_idx + test_iter
            if i == 0:
                print triplet_cnts[test_beg_idx:test_end_idx]
            test_triplet_cnts.append(int(np.array(triplet_cnts[test_beg_idx:test_end_idx]).mean()))

            for j in range(train_cnt):
                train_beg_idx = test_end_idx + j*display
                train_end_idx = train_beg_idx + display
                train_triplet_cnts.append(int(np.array(triplet_cnts[train_beg_idx:train_end_idx]).mean()))

        idx_train_triplets = [idx*display for idx in xrange(len(train_triplet_cnts))]
        idx_test_triplets = [idx*cnt for idx in xrange(len(test_triplet_cnts))]

        plt.figure(1)
        plt.plot(idx_train_triplets, train_triplet_cnts)
        plt.plot(idx_test_triplets, test_triplet_cnts)
        plt.xlabel('iterations')
        plt.ylabel('triplets number')
        plt.title('Train & Test Triplet Number')
        plt.legend(['Train Triplet', 'Test Triplet'])
        # plt.show()

    plt.figure(2)
    plt.plot(idx_train_losses, train_losses)
    plt.plot(idx_test_losses, test_losses)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('Train & Test Loss Curve')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()

if __name__ == '__main__':
    main()

