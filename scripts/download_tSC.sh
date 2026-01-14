#!/bin/sh

dataset_root=${1:-data}

cd ${dataset_root}
gdown --id 17prYkldYb3w3Pyg3Pm77-VnE6nkD5jzG
gdown --id 19ZnkM4vjApCZipd7xQ1ESlOi5oBVrlFL
unzip tSC.zip
unzip sSC.zip