#!/bin/sh

git clone https://github.com/cheoljun95/sdhubert.git src/sdhubert
git clone https://github.com/jasonppy/syllable-discovery.git src/vghubert
git clone https://github.com/Berkeley-Speech-Group/sylber.git src/sylber
git clone https://github.com/AlanBaade/SyllableLM.git src/SyllableLM
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo src/utmos
git clone https://github.com/facebookresearch/textlesslib.git src/textlesslib

cd src/sdhubert
git checkout ecb6469
cd -

cd src/sdhubert/mincut
python setup.py build_ext --inplace
cd -

patch src/sdhubert/extract_segments.py src/patch/sdhubert_extract_segments.patch
patch src/sdhubert/utils/misc.py src/patch/sdhubert_utils_misc.patch
patch src/utmos/lightning_module.py src/patch/utmos_lightning_module.patch