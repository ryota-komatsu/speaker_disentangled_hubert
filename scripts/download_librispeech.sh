#!/bin/sh

dataset_root=${1:-data}

wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/train-other-500.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/dev-clean.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/dev-other.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/test-clean.tar.gz
wget -t 0 -c -P ${dataset_root}/LibriSpeech https://www.openslr.org/resources/12/test-other.tar.gz

tar zxvf ${dataset_root}/LibriSpeech/train-clean-100.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/train-clean-360.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/train-other-500.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/dev-clean.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/dev-other.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/test-clean.tar.gz -C ${dataset_root}
tar zxvf ${dataset_root}/LibriSpeech/test-other.tar.gz -C ${dataset_root}