# -*- coding: utf-8 -*-

import numpy as np
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
    #テキストに偽陰性率を出力
    f = open('./result/{}/mel_FalseNegativeRate.txt'.format(state), 'w')
    
    #aの値を持つnum配列
    num_data = np.arange(0, 2, 0.1)
    
    for i, num in enumerate(num_data): 

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


        #閾値を出力
        hist, edge = np.histogram(D, bins=bin_num, range=(min_range, max_range))
        hist_total=hist.sum()
        hist_sum = 0.0
        for hist_num, edge_num in zip(hist, edge):
            hist_sum += hist_num
            if hist_sum/hist_total >= 0.8:
                thre = edge_num
                break


        #出力
        #pyplot.hist(D, bins=130, alpha=a, color='red', histtype='stepfilled')
        pyplot.hist(D, bins=bin_num, alpha=a, color='red', range=(min_range, max_range), histtype='stepfilled')
        #pyplot.hist(D, bins=130, alpha=a, color='red', range=(0, 14), histtype='step')
        
        
        del D
        del mahala
        del mahala_deno
        del mahala_nume
        del D_deno
        del D_nume





        #ヒストグラムを重ねて書くか
        pyplot.hold(True)




        #異常音のマハラノビス距離
        mahala, D = np.array([]), np.array([])
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
        D = D_nume / D_deno



        #誤受理率, 偽陰性率＝異常音だけど正常音だと認識された確率
        hist_abnomal, edge_abnomal = np.histogram(D, bins=bin_num, range=(min_range, max_range))
        hist_sum = 0.0
        for hist_num, edge_num in zip(hist_abnomal, edge_abnomal):
            if edge_num <= thre:
                hist_sum += hist_num
            else:
                break
        #偽陰性率
        false_negative_rate = hist_sum / test_abnomal.shape[0]
        #出力
        f.write('{0}\t{1}\n'.format(num, false_negative_rate))
        #閾値の出力
        pyplot.plot([thre, thre], [0, np.max(hist)+100], "black", linestyle='dashed', linewidth = 3.0)
        pyplot.yticks( np.arange(0, np.max(hist)+100, 100) )
        pyplot.ylim([0, np.max(hist)+50])

        #pyplot.step(edge[:-1], hist, where='post', color='green')
        
        del hist_abnomal
        del edge_abnomal



        #出力
        #pyplot.hist(D, bins=130, alpha=a, color='blue', histtype='stepfilled')
        pyplot.hist(D, bins=bin_num, alpha=a, color='blue', range=(min_range, max_range), histtype='stepfilled')
        #pyplot.hist(D, bins=130, alpha=a, color='blue', range=(0, 15), histtype='step')
        #pyplot.savefig('./result/hist_propose_abnomal_a={}.png'.format(num), dpi=300)
        #pyplot.xlim([0, 7])
        #pyplot.ylim([0, 600])
        pyplot.savefig('./result/{0}/{0}_propose_a={1}.png'.format(state, num), dpi=300)

    
        del D
        del mahala
        del mahala_deno
        del mahala_nume
        del D_deno
        del D_nume

        del hist
        del edge
        del hist_total
        del thre


        pyplot.hold(False)
    
    f.close()
    
 
if __name__ == '__main__':
    main()
