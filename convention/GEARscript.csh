#!/bin/csh -f


#���s���@
#Linux zsh�̏ꍇ
#zsh GEARscript.csh


# train feature extraction --------------------------------------------------------
# �P���p�̐���Ȏ��ԉ���������ʁi�ΐ��p���[�X�y�N�g���j�𒊏o

foreach state ( normal )
    python make_pkl.py train $state 2048 960 480 16 48000
end


# test feature extraction --------------------------------------------------------
# �]���p�̎��ԉ���������ʁi�ΐ��p���[�X�y�N�g���j�𒊏o

foreach state ( normal 0_abnormal 1_abnormal )
    python make_pkl.py test $state 2048 960 480 16 48000
end


# pickle�t�@�C����ǂݍ��݃��f���𐶐�--------------------------------------------

foreach state ( normal 0_abnormal 1_abnormal )
    python mahara_norm_weight.py test $state 
    python experiment.py 0 21 0.7 182  test $state   #�͈�/(15/130)��bin��������
end


# PR�Ȑ���`��--------------------------------------------------------------------
# python hist_PR.py 0 21 0.7 182  test 0_abnormal alpha
# alpha�F��Lexperiment.py�ōŗǎ��ʌ��ʂ̏d��

python hist_PR.py 0 21 0.7 182  test 0_abnormal 1.0
python hist_PR.py 0 21 0.7 182  test 1_abnormal 1.4




