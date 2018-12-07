#!/bin/csh -f


#実行方法
#Linux zshの場合
#zsh GEARscript.csh


# train feature extraction --------------------------------------------------------
# 訓練用の正常な歯車音から特徴量（対数パワースペクトル）を抽出

foreach state ( normal )
    python make_pkl.py train $state 2048 960 480 16 48000
end


# test feature extraction --------------------------------------------------------
# 評価用の歯車音から特徴量（対数パワースペクトル）を抽出

foreach state ( normal 0_abnormal 1_abnormal )
    python make_pkl.py test $state 2048 960 480 16 48000
end


# pickleファイルを読み込みモデルを生成--------------------------------------------

foreach state ( normal 0_abnormal 1_abnormal )
    python mahara_norm_weight.py test $state 
    python experiment.py 0 21 0.7 182  test $state   #範囲/(15/130)でbin数を決定
end


# PR曲線を描画--------------------------------------------------------------------
# python hist_PR.py 0 21 0.7 182  test 0_abnormal alpha
# alpha：上記experiment.pyで最良識別結果の重み

python hist_PR.py 0 21 0.7 182  test 0_abnormal 1.0
python hist_PR.py 0 21 0.7 182  test 1_abnormal 1.4




