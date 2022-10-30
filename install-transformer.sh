#!/bin/bash

# download repository by git
git clone https://github.com/WeilabMSU/PretrainModels.git

cd PretrainModels/bt_pro
python setup.py build_ext --inplace
mv ./bt_pro/fairseq/data/* ./fairseq/data/

cd ..
# Pre-trained model
wget https://weilab.math.msu.edu/Downloads/chembl_pubchem_zinc_models.zip
unzip chembl_pubchem_zinc_models.zip

cp ../generate_bt_fps_mean.py bt_pro
