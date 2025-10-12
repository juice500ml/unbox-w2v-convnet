echo "Downloading TIMIT..."
git clone --branch bugfix_tab --single-branch https://github.com/juice500ml/ldc_downloader.git
./ldc_downloader/download-ldc-corpora LDC93S1
tar -xvzf timit_LDC93S1.tgz

echo "Downloading VoxAngeles..."
git clone --branch main --single-branch https://github.com/pacscilab/voxangeles.git
for zipfile in voxangeles/data/audited_aligned/*.zip; do
    unzip -q "$zipfile" -d "$(dirname "$zipfile")" && rm "$zipfile"
done
