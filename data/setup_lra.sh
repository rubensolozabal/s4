# wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar xvf lra_release.gz
mv lra_release/lra_release/listops-1000 listops
mv lra_release/lra_release/tsv_data aan
mkdir pathfinder
mv lra_release/lra_release/pathfinder* pathfinder/
rm -r lra_release