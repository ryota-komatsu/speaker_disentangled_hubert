#!/bin/sh

# setup
# conda create -y -n sdhubert python=3.10.14 pip=24.0
# conda activate sdhubert
# pip install -r requirements/requirements_for_sdhubert.txt

dataset_root=${1:-data}

if [ ! -d models/sdhubert_base ]
then
    echo download SD-HuBERT models from the following links and place them under models/sdhubert_base/
    echo https://drive.google.com/file/d/1u2jTdAck8qD6ZEb5bqHfvUNsN-9DgGfg/view?usp=drive_link
    echo https://drive.google.com/file/d/14zdEttya2X8PdjDMUt4lyHWOOY2OS3Zr/view?usp=drive_link
    echo https://drive.google.com/file/d/19XisepDAfULOKFY147RDYT5UAk2ZnCr-/view?usp=drive_link
    exit 1
fi

cd src/sdhubert

python extract_segments.py \
    --ckpt_path ../../models/sdhubert_base/sdhubert_base.pt \
    --librispeech_dataroot ${dataset_root}/LibriSpeech \
    --save_dir ../../segments

cd -

python main.py --config configs/sdhubert.yaml