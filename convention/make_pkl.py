#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import argparse
import commands
import time
import sys
import cPickle as pickle
import os.path
import tool


#配列argsに引数を格納
args = sys.argv

# 引数:python make_pkl.py train　normal 2048 960 480 16 48000
data_type = args[1] #trainか、testか
state = args[2] #normalかabnormalか
N_FFT = int(args[3]) #FFT長
frameLen = int(args[4]) #フレーム分析幅 [point] 48000Hzなら20[ms]で960点
Shift = int(args[5]) #オーバーラップ数 [point]  10[ms]より、480点
bit = int(args[6]) #ビット数
frequency = int(args[7]) #周波数


    
#pklファイルを作るための関数
def write_pkl(data, data_type, state):
    #例.  ./pickle/train/normalのフォルダを作成
    commands.getoutput('mkdir -p ./pickle/%s/%s/' % (data_type, state))
    #例.  ./pickle/train/normal/train_normal.pklを開く
    with open('./pickle/{0}/{1}/{0}_{1}.pkl'.format(data_type, state), 'wb') as f:
        pickle.dump(data, f, protocol=-1)
    f.close() 



if __name__ == '__main__':


    print('############################################################')
    print('sound/{0}/{1}　のpklファイルを作成します'.format(data_type, state)) 


    #PKLファイルがあるならこのプログラムを実行しない
    if os.path.exists('./pickle/train/normal/train_normal.pkl'):
        print('-----------------------------------')
        print('{}のPKLは存在します'.format('Train'))
        if os.path.exists('./pickle/test/{0}/{1}_{2}.pkl'.format(state, data_type, state)):	
            print('{0}_{1}のPKLは存在します'.format(data_type, state))
            sys.exit()


    # ---------------------------------------------------------------------------------------------------- #
    print('  --- feature extraction --------')
    print('  type: {}'.format(data_type)) #trainかtestか
    print('  state: {}'.format(state)) #normalかabnormalか
    print('  FFT length: {}'.format(N_FFT))
    print('  frameLen: {}'.format(frameLen)) #960
    print('  Shift: {}'.format(Shift)) #480
    print('  bit: {}'.format(bit)) #16bit
    print('  frequency: {}'.format(frequency)) #周波数
    print('  -----------------------------------')


    # ---------------------------------------------------------------------------------------------------- #

    #音源を扱う
    x = np.array([])
    print('    {0}-{1} data loading...'.format(data_type, state))
    files = sorted(glob.glob('./sound/{0}/{1}/*raw'.format(data_type, state)))


    #配列iはファイル数、配列filePathは./sound/data_type(trainやtestなど)/$state(normal等)/a.raw
    for i, filePath in enumerate(files): 
    
        sp = tool.spectrum_envelope(filePath, N_FFT, frameLen, Shift, bit, frequency, i, data_type, state)


        if i==0: # 初回のみコピー，その他は同じ変数にスタック
            x = sp
        else:
            x = np.vstack((x, sp))

        print('      {0} done (size = {1},  loop = {2})' .format( filePath, len(sp), i ))
        
        del sp


    print('-----------------------------------')
    print('    {0}-{1} done'.format(data_type, state) )
    print('    feature_size: {} '.format(x.shape))


    # pklファイルに出力	
    write_pkl(x, data_type, state)

    del x



