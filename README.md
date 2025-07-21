# Industrial Product Anomaly Detection

## Code Usage
### 1) Get start

* Python 3.9.x
* CUDA 11.1 or *higher*
* NVIDIA RTX 3090
* Torch 1.8.0 or *higher*

**Create a python env using conda**
```bash
conda create -n hetmm python=3.9 -y
conda activate hetmm
```

**Install the required libraries**
```bash
bash setup.sh
```

### 2) Template Generation
**Original template set on MVTec AD:**
```bash
python run.py --mode temp --ttype ALL --dataset MVTec_AD --datapath <data_path>
```
**Tiny set formed by PTS (60 sheets) on MVTec AD:**
```bash
python run.py --mode temp --ttype PTS --tsize 60 --dataset MVTec_AD --datapath <data_path>
```
Since generating pixel-level OPTICS clusters is time-consuming, you can download the "*template*" folder from [Google Drive](https://drive.google.com/drive/folders/1c4XvmugX-ryP168bDMFcScdiYWgYktlu?usp=drive_link) / [Baidu Cloud](https://pan.baidu.com/s/1HH_3FQo1K72HbUvZpfylxw?pwd=eeg9) and copy it into our main folder as:
```
HETMM/
    ├── configs/
    ├── template/
    ├── src/
    ├── run.py
    └── ...
```

### 3) Anomaly Prediction
**Original template set on MVTec AD:**
```bash
python run.py --mode test --ttype ALL --dataset MVTec_AD --datapath <data_path>
```
**Tiny set formed by PTS (60 sheets) on MVTec AD:**
```bash
python run.py --mode test --ttype PTS --tsize 60 --dataset MVTec_AD --datapath <data_path>
```
Please see "*run.sh*" and "*run.py*" for more details.

## Acknowledgement
The authors would like to express their gratitude to the Bandung Institute of Technology for
providing the Tambora Server as computational and storage resources for this research.

Also thank you to Zixuan Chen et. al. for the inspiration to be the baseline model. https://github.com/NarcissusEx/HETMM

And Hanxi Li for the inspiration of the FEB model. https://github.com/flyinghu123/cpr

## References
Chen, Z., Xie, X., Yang, L., and Lai, J. H. (2025): Hard-normal example-aware template mutual matching for industrial anomaly detection, International Journal of Computer Vision, 133, 2927–2949. https://doi.org/10.1007/s11263-024-02323-0

Li, H., Hu, J., Li, B., Chen, H., Zheng, Y., and Shen, C. (2024): Target before shooting: accurate anomaly detection and localization under one millisecond via cascade patch retrieval, IEEE Transactions on Image Processing, 33, 5606-5621. https://doi.org/10.1109/TIP.2024.3448263
