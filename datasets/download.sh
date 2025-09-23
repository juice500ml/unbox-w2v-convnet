git clone --branch fix_macosx --single-branch https://github.com/juice500ml/voxangeles.git
for zipfile in voxangeles/data/audited_aligned/*.zip; do
    unzip -q "$zipfile" -d "$(dirname "$zipfile")" && rm "$zipfile"
done
