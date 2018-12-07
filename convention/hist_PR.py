# -*- coding: utf-8 -*-
#本コードは旧バージョン(1.11.0)以前を意識して書いたコード
#コピーするコード
#scp gear_test.py is0269rx@172.25.13.49:/data/is0269rx/traning
#サーバに接続するコード
#ssh is0269rx@172.25.13.49 
import numpy as np
#DNNサーバのバージョンは1.8
import sys
import cPickle as pickle
import commands
import glob
import os.path
import sys
import argparse
from matplotlib import pyplot

args = sys.argv
min_range = int(args[1])
max_range = int(args[2])
a = float(args[3])
bin_num = int(args[4])
data_type = args[5]
state = args[6]
alpha_num = float(args[7])



#pklファイルを開く関数
def read_pkl(data_type, state):

    # ./pickle/train/nomal/train_normal.pkl
    with open('./pickle/{0}/{1}/{0}_{1}.pkl'.format( data_type, state ), 'rb') as f:
        data = pickle.load(f)
    f.close()

    return data




def main():
    
    print('############################')
    print('マハラノビス距離({})のヒストグラムを作成します'.format(state))
    

    ##################################
    #訓練データ、テストデータの取得
    ##################################


    #訓練用正常音を読み込む。
    train_nomal = np.array([])

    print('  -----------------------------------')
    print('  Training data loading...')
    print('  -----------------------------------')

    train_nomal = read_pkl( 'train', 'nomal' )
    print('     "{0}" normaly done. size:{1}'.format('Train', train_nomal.shape))

    train_nomal = train_nomal.astype(np.float32)



    #テスト用正常音を読み込む。
    test_nomal = np.array([])

    print('  -----------------------------------')
    print('  Test (nomal) data loading...')
    print('  -----------------------------------')

    test_nomal = read_pkl( 'test', 'nomal')
    print('     "{0}" normaly done. size:{1}'.format('test', test_nomal.shape))

    test_nomal = test_nomal.astype(np.float32)


    #テスト用異常音を読み込む。
    test_abnomal = np.array([])

    print('  -----------------------------------')
    print('  Test (abnomal) data loading...')
    print('  -----------------------------------')

    test_abnomal = read_pkl( data_type, state )
    print('     "{0}" abnormaly done. size:{1}'.format('test', test_abnomal.shape))

    test_abnomal = test_abnomal.astype(np.float32)


	#################################
	####　対数パワースペクトルの平均と分散　#####
	#################################

    
    #初期化
    x_mean, x_var = np.array([]), np.array([])
    
    #列ごとの平均をとった
    x_mean = train_nomal.mean( axis = 0 )
    #分散
    x_var = train_nomal.var( axis = 0 )

    #print('-----------------------------------')
    #print(x_mean)
    #print(x_var)

    #print('  x_mean feature_size: {} '.format( x_mean.shape))
    #print('  x_var  feature_size: {} '.format( x_var.shape))



    #################################
    ######　　マハラノビス距離の算出       #######
    #################################
    
    #結果をテキストに出力
    commands.getoutput('mkdir -p ./result/{0}'.format(state))
    
    #テキストにPR曲線を出力
    f = open('./result/{0}/PrecisionRecall_a={1}.txt'.format(state, alpha_num), 'w')

    num=alpha_num



    #正常音のマハラノビス距離
    mahala, D = np.array([]), np.array([])
    mahala_deno, mahala_nume = np.array([]), np.array([])
    D_deno, D_nume = np.array([]), np.array([])
    
    mahala =  ( (test_nomal - x_mean) ** 2 ) / x_var
    np.sqrt(mahala)
        
    #累乗(=分子2.4乗、分母1.4乗)を計算
    mahala_nume = mahala ** (num+1)
    mahala_deno = mahala ** num
    #行ごとの和をとって比をとる
    D_nume =  np.sum(mahala_nume, axis=1)
    D_deno =  np.sum(mahala_deno, axis=1)
    D = D_nume / D_deno
    
    del mahala
    del mahala_deno
    del mahala_nume
    del D_deno
    del D_nume



    #異常音のマハラノビス距離
    mahala, D2 = np.array([]), np.array([])
    mahala_deno, mahala_nume = np.array([]), np.array([])
    D_deno, D_nume = np.array([]), np.array([])


    mahala =  ( (test_abnomal - x_mean) ** 2 ) / x_var
    np.sqrt(mahala)
    
    
    #累乗(=分子2.4乗、分母1.4乗)を計算
    mahala_nume = mahala ** (num+1)
    mahala_deno = mahala ** num     
    #行ごとの和をとって比をとる
    D_nume =  np.sum(mahala_nume, axis=1)
    D_deno =  np.sum(mahala_deno, axis=1)
    D2 = D_nume / D_deno




    #閾値の値を持つnum配列
    num_data = np.arange(0, 1, 0.01)

    #閾値を変えるるーぷ
    for i, num in enumerate(num_data): 
        TP=0.0
        TN=0.0
        FP=0.0
        Recall=0.0
        Precision=0.0

        #閾値を出力
        hist, edge = np.histogram(D, bins=bin_num, range=(min_range, max_range))
        hist_total=hist.sum()
        hist_sum = 0.0
        for hist_num, edge_num in zip(hist, edge):
            hist_sum += hist_num
            if hist_sum/hist_total >= num:
                thre = edge_num
                break


        #PR曲線
        hist_abnomal, edge_abnomal = np.histogram(D2, bins=bin_num, range=(min_range, max_range))
        hist_sum = 0.0

        #偽陰性を求める。
        for hist_num, edge_num in zip(hist_abnomal, edge_abnomal):
            if edge_num <= thre:
                hist_sum += hist_num
            else:
                break

        #真陽性を求める。
        TP = test_abnomal.shape[0] - hist_sum

        #真陰性を求める。(正常音から)
        for hist_num, edge_num in zip(hist, edge):
            if edge_num <= thre:
                TN += hist_num
            else:
                break

        #偽陽性を求める。
        FP = test_nomal.shape[0] - TN
        
        #適合率
        Precision = TP / (TP + FP)
        
        #再現率
        Recall = TP / test_abnomal.shape[0]
        
        #出力
        f.write('{0}\t{1}\n'.format(Recall, Precision))

    
    del hist_abnomal
    del edge_abnomal


    del D
    del D2
    del mahala
    del mahala_deno
    del mahala_nume
    del D_deno
    del D_nume

    del hist
    del edge
    del hist_total
    del thre

    f.close()
    
 
if __name__ == '__main__':
    main()
