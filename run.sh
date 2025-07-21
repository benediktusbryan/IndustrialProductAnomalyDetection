dataset=MVTec_AD
datapath=/mvtec ad dataset path
tt=PTS
ts=60

# Generating template
# Original template
python run.py --mode temp --ttype ALL --dataset $dataset --datapath $datapath
# PTS template
python run.py --mode temp --ttype $tt --tsize $ts --dataset $dataset --datapath $datapath

# Anomaly detection and localization
# Original template
python run.py --mode test --ttype ALL --dataset $dataset --datapath $datapath --save_map
# PTS template
python run.py --mode test --ttype $tt --tsize $ts --dataset $dataset --datapath $datapath --save_map