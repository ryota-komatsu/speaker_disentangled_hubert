#!/bin/sh

cd src/speaker_disentangled_hubert/mincut
python setup.py build_ext --inplace
cd -

git clone https://github.com/cheoljun95/sdhubert.git src/sdhubert
git clone https://github.com/jasonppy/syllable-discovery.git src/vghubert

cd src/sdhubert
git checkout ecb6469
cd -

cd src/sdhubert/mincut
python setup.py build_ext --inplace
cd -

patch src/sdhubert/extract_segments.py src/patch/sdhubert_extract_segments.patch
patch src/sdhubert/utils/misc.py src/patch/sdhubert_utils_misc.patch