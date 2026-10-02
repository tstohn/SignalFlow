


###############################
#    ARC DATA
###############################


###############################
#    Virtual Cell Challanege 2025
###############################

#make fodlers for data
mkdir -p data
mkdir -p data/VCC25
mkdir -p data/TAHOE100M
mkdir -p data/REPLOG
mkdir -p data/NADIG

#WE PAY if we download more than 2TB, just check what is in folders before dwonload
gsutil ls gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/

#the VCC data contain test/ train and validation data
gcloud storage cp -r gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train ./data/VCC25/
gcloud storage cp -r gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/test ./data/VCC25/
gcloud storage cp -r gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/validation ./data/VCC25/


###############################
#    TAHOE 100
###############################

gsutil du -sh gs://arc-institute-virtual-cell-atlas/tahoe100M/2025-02-25/h5ad/ 
gsutil ls gs://arc-institute-virtual-cell-atlas/tahoe100M/2025-02-25/

gcloud storage cp -r gs://arc-institute-virtual-cell-atlas/tahoe100M/2025-02-25/ ./data/TAHOE100M/


###############################
#    scBase dataset
###############################

gsutil du -sh gs://arc-institute-virtual-cell-atlas/scbasecount/


###############################
#    REPLOG DATASET (scPerturb-seq)
###############################

#PAPER: https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/#S11
#DOWNLOAD LINKS with INFO: https://gwps.wi.mit.edu/
#FIGSHARE DOWNLOAD PAGE (download only RAW single-cell data): 
#https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387?file=35775606

# 1. K562_gwps_raw_singlecell_01.h5ad (~61.31 GB)
cd data/REPLOG
curl -C - -L -H "User-Agent: Mozilla/5.0" "https://api.figshare.com/v2/file/download/35775507" -o K562_gwps_raw_singlecell_01.h5ad
# 2. K562_essential_raw_singlecell_01.h5ad (~9.93 GB)
curl -C - -L -H "User-Agent: Mozilla/5.0" "https://api.figshare.com/v2/file/download/35773219" -o K562_essential_raw_singlecell_01.h5ad
# 3. rpe1_raw_singlecell_01.h5ad (~8.1 GB)
curl -C - -L -H "User-Agent: Mozilla/5.0" "https://api.figshare.com/v2/file/download/35775606" -o rpe1_raw_singlecell_01.h5ad

###############################
#    NADIG et al
###############################
cd ../NADIG
# 1. Download HepG2 raw single-cell data (5.2 GB)
curl -C - -L -o "hepg2_raw_singlecell.h5ad" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fhepg2%5Fraw%5Fsinglecell%5F01%2Eh5ad"
# 2. Download Jurkat raw single-cell data
curl -C - -L -o "jurkat_raw_singlecell.h5ad" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE264667&format=file&file=GSE264667%5Fjurkat%5Fraw%5Fsinglecell%5F01%2Eh5ad"


#############################
# PROCESS & COMPRESS
#############################

#run pytoh script
tar -czvf REPLOG.tar.gz REPLOG
tar -tzvf REPLOG.tar.gz > /dev/null && echo "Archive OK" && rm -rf REPLOG

tar -czvf NADIG.tar.gz NADIG
tar -tzvf NADIG.tar.gz > /dev/null && echo "Archive OK" && rm -rf NADIG

tar -czvf VCC25.tar.gz VCC25
tar -tzvf VCC25.tar.gz > /dev/null && echo "Archive OK" && rm -rf VCC25




cd ../..
