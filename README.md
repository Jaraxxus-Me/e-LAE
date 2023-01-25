# eLAE-toolkit
## Install 


```bash
pip install -r requirements.txt
cd pysot/utils/
python setup.py build_ext --inplace
# if you need to draw graph, you need latex installed on your system
```

## Usage

### 1. e-LAE 
```bash
python bin/eval.py --dataset_dir /path/to/UAV123  \	# dataset path
--dataset UAV123 \	# dataset
--tracker_result_dir /path/to/results_eLAE/UAV123 \ # result path
--trackers ATOM DaSiamRPN DiMP18 DiMP50 HiFT PrDiMP SiamAPN SiamAPN++ SiamBAN \
SiamCAR SiamFC++_Alex SiamFC++_Google SiamGAT SiamMask SiamRPN_Alex SiamRPN++_Mob \
SiamRPN++_Res TrDiMP # trackers
```
### 2. sigma=0

```bash
python bin/eval.py --dataset_dir /path/to/DTB70  \ # dataset path
--dataset DTB70 \ # dataset
--tracker_result_dir /path/to/results_rt/DTB70 \ # result path
--eta 0.0 \
--trackers RPN_Mob_M RPN_Mob_V RPN_Mob_MV # trackers
```
