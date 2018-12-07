#coding:utf-8

"""

    N:  フーリエ点数 2048
    frameLen:  標本数( t[ms]/(1/x[kHz]) )  960
    shift:  オーバーラップ点数  480
    bit:  bit数
    frequency: 周波数 48000Hz
 
"""
import numpy as np
import sys
import wave
import struct
from matplotlib import pyplot
import commands


args = sys.argv


def openfile(filename):

    #ファイル指定
    fb = open(filename, 'rb')
    #ファイル読み込み
    fp = np.fromfile(fb, np.int16, -1)
    #型変換
    fp = fp.astype(np.float32)
    #予めファイルをクローズ
    fb.close()

    return fp


#正規化
def zscore(x):
    xmean = x.mean(keepdims=True)
    xstd  = np.std(x, keepdims=True)
    zscore = (x-xmean)/xstd

    return zscore


#周波数軸上での出力
def freq_output(freq, x, num, state, data_type):
    pyplot.plot(freq, x, color='red')
    pyplot.xlim([0, 24000])
    pyplot.ylim([10, 110])
    pyplot.xlabel("Frequency [Hz]")
    pyplot.ylabel("Power [dB]")
    pyplot.savefig('./import/result/SpeEm_{0}_{1}_{2}.png'.format(num, data_type, state), dpi=300)

    return 0


def spectrum_envelope(filePath, N, frameLen, shift, bit, frequency, num, data_type, state):
    
    """
    filePath: ファイル名
    N:  フーリエ点数 2048
    frameLen:  標本数( t[ms]/(1/x[kHz]) )  960
    shift:  オーバーラップ点数  480
    bit:  bit数
    frequency: 周波数 48000Hz
    """
    #結果をテキストに出力
    commands.getoutput('mkdir -p ./import/result/')
    
    #周波数間隔
    #t = float(1)/frequency
    
    #ファイル読み込み
    x = openfile(filePath)
    
    #正規化
    zx = zscore(x)
    
    #フレーム毎に処理
    for i in range(0, zx.shape[0]-shift, shift):

        #ファイルの初めからフレーム数分、値を参照
        wave = zx[ i : i+frameLen]
    
        # ハニング窓をかける
        hanningWindow = np.hanning(wave.shape[0])
        wave = wave * hanningWindow
        
        # 離散フーリエ変換
        dft = np.fft.fft(wave, N)
    
        #周波数軸を算出
        freq = np.fft.fftfreq(N, 1.0/frequency)
        freq = np.delete(freq, range(N/2, N), 0)
    
        # パワースペクトル
        Pdft = np.abs(dft) ** 2
        
        # 対数パワースペクトル
        PdftLog = 10 * np.log10(Pdft)
        cutPdfLog = np.delete(PdftLog, range(N/2, N), 0)
        #対数パワースペクトルの出力
        #pyplot.plot(freq, cutPdfLog, color='black') 

        
        """
        #ヒストグラムを重ねて書くか
        #pyplot.hold(True)
        
        #周波数軸で出力
        #freq_output(freq, cutdftSpc, num, state, data_type)

        #ヒストグラムを重ねて書くか
        #pyplot.hold(False)
        """
        
        #軸の変換を行う
        cutPdfLog = cutPdfLog.reshape((1,cutPdfLog.shape[0]))
        
        if i==0: # 初回のみコピー，その他は同じ変数にスタック
            output = cutPdfLog
        else:
            output = np.append(output, cutPdfLog, axis=0)
    
        del cutPdfLog
        
    """    
    print('done')
    print('size={}'.format(output.shape))
    """
    
    return output






