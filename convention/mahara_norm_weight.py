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
N = 1024


#pklファイルを開く関数
def read_pkl(input_type, folder, file_name):

    # ./pickle/train/nomal/train_normal.pkl
    with open('./pickle/%s/%s/%s.pkl' % ( input_type, folder, file_name ), 'rb') as f:
        data = pickle.load(f)
    f.close()

    return data



def main():
    
    print('############################')
    print('論文のように重みをつけたマハラノビス距離を出力')
    

    ##################################
    #訓練データ、テストデータの取得
    ##################################


    #訓練用正常音を読み込む。
    train_nomal = np.array([])

    print('  -----------------------------------')
    print('  Training data loading...')
    print('  -----------------------------------')

    train_nomal = read_pkl( 'train', 'nomal', 'train_nomal' )
    print('     "{0}" normaly done. size:{1}'.format('Train', train_nomal.shape))

    train_nomal = train_nomal.astype(np.float32)


    #テスト用正常音を読み込む。
    test_nomal = np.array([])

    print('  -----------------------------------')
    print('  Test (nomal) data loading...')
    print('  -----------------------------------')

    test_nomal = read_pkl( 'test', 'nomal', 'test_nomal' )
    print('     "{0}" normaly done. size:{1}'.format('test', test_nomal.shape))

    test_nomal = test_nomal.astype(np.float32)


    #テスト用異常音を読み込む。
    test_abnomal = np.array([])

    print('  -----------------------------------')
    print('  Test (abnomal) data loading...')
    print('  -----------------------------------')

    test_abnomal = read_pkl( 'test', 'abnomal', 'test_abnomal' )
    print('     "{0}" abnormaly done. size:{1}'.format('test', test_abnomal.shape))

    test_abnomal = test_abnomal.astype(np.float32)



	#################################
	#対数パワースペクトルの平均と分散
	#################################

    #print('-----------------------------------')
    #print(' 平均と分散をとる')
    
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
    commands.getoutput('mkdir -p ./result/')
    #ヒストグラムを重ねて書くか
    #pyplot.hold(True)
    pyplot.hold(False)


    ##############
    #重みa=0の場合
    ##############

    mahala, D = np.array([]), np.array([])
    mahala =  (test_abnomal - x_mean) * (1 / x_var ) * (test_abnomal - x_mean)
    
    #異常音のマハラノビス距離に重み(a=0)をつける
    mahala_w = mahala
    
    #異常音の各周波数ごとにマハラノビス距離の平均をとる
    mahala_norm = np.array([])
    mahala_norm = np.sum(mahala_w, axis=0)
    mahala_norm = mahala_norm / mahala_w.shape[0]
    """
    #正規化
    xmean = mahala_norm.mean(keepdims=True)
    xstd  = np.std(mahala_norm, keepdims=True)
    zscore = (mahala_norm - xmean)/xstd
    """
    """
    #異常音の各周波数ごとのマハラノビス距離の分散をとる
    mahala_v = np.array([])
    mahala_v = np.std( mahala, axis = 0 )
    """
    #周波数軸を算出
    freq = np.fft.fftfreq(2048, 1.0/48000)
    freq = np.delete(freq, range(1024, 2048), 0)
    #出力
    #pyplot.errorbar(freq,mahala_norm,yerr=mahala_v,ecolor='blue')
    #pyplot.plot(freq, zscore, color='blue')
    pyplot.plot(freq, mahala_norm, color='black')
    pyplot.xticks( np.arange(0, 25000, 4000) )
    pyplot.xlabel("Frequency [Hz]")
    #pyplot.ylabel("Power [dB]")
    pyplot.xlim([0, 24000])
    pyplot.ylim([0, 9])
    pyplot.savefig("./result/weight_a=0_abnomal_mahara.png", dpi=300)
    
    
    del mahala
    del mahala_norm
    del mahala_w
    """
    del mahala_v
    del xmean
    del xstd
    del zscore
    """
    
    
    ##############
    #重みa=1.4の場合
    ##############

    mahala, D = np.array([]), np.array([])
    mahala =  (test_abnomal - x_mean) * (1 / x_var ) * (test_abnomal - x_mean)
    
    #異常音のマハラノビス距離に重み(a=1.4)をつける
    mahala_w = mahala ** 1.4
    
    #異常音の各周波数ごとにマハラノビス距離の平均をとる
    mahala_norm = np.array([])
    mahala_norm = np.sum(mahala_w, axis=0)
    mahala_norm = mahala_norm / mahala_w.shape[0]
    """
    #正規化
    xmean = mahala_norm.mean(keepdims=True)
    xstd  = np.std(mahala_norm, keepdims=True)
    zscore = (mahala_norm - xmean)/xstd
    """
    """
    #異常音の各周波数ごとのマハラノビス距離の分散をとる
    mahala_v = np.array([])
    mahala_v = np.std( mahala, axis = 0 )
    """
    #周波数軸を算出
    freq = np.fft.fftfreq(2048, 1.0/48000)
    freq = np.delete(freq, range(1024, 2048), 0)
    #出力
    #pyplot.errorbar(freq,mahala_norm,yerr=mahala_v,ecolor='blue')
    #pyplot.plot(freq, zscore, color='blue')
    pyplot.plot(freq, mahala_norm, color='black')
    pyplot.xticks( np.arange(0, 25000, 4000) )
    pyplot.xlabel("Frequency [Hz]")
    #pyplot.ylabel("Power [dB]")
    pyplot.xlim([0, 24000])
    pyplot.ylim([0, 9])
    pyplot.savefig("./result/weight_a=1.4_abnomal_mahara.png", dpi=300)
    
    
    del mahala
    del mahala_norm
    """
    del mahala_v
    del xmean
    del xstd
    del zscore
    """
 
 
if __name__ == '__main__':
    main()
