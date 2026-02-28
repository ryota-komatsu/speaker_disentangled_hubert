#!/bin/sh

# setup
# conda create -y -n sdhubert python=3.10.14 pip=24.0
# conda activate sdhubert
# pip install -r requirements/sdhubert.txt

dataset_root=${1:-data}

if [ ! -d models/sdhubert_base ]
then
    mkdir models/sdhubert_base
    cd models/sdhubert_base
    gdown --id 1u2jTdAck8qD6ZEb5bqHfvUNsN-9DgGfg
    gdown --id 14zdEttya2X8PdjDMUt4lyHWOOY2OS3Zr
    gdown --id 19XisepDAfULOKFY147RDYT5UAk2ZnCr-
    cd -
fi

cd src/sdhubert

python extract_segments.py \
    --ckpt_path ../../models/sdhubert_base/sdhubert_base.pt \
    --librispeech_dataroot ${dataset_root}/LibriSpeech \
    --save_dir ../../segments

cd -

python main_speech2unit.py evaluate --config configs/speech2unit/sdhubert.yaml