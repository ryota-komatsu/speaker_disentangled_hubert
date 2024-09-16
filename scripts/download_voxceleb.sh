#!/bin/sh

dataset_root=${1:-data}

wget -t 0 -c -P ${dataset_root}/VoxCeleb1 https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip
wget -t 0 -c -P ${dataset_root}/VoxCeleb1 https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip

cd ${dataset_root}/VoxCeleb1

unzip -q vox1_dev_wav.zip
unzip -q vox1_test_wav.zip