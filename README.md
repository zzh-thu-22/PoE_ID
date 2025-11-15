# Option-ID Based Elimination For Multiple Choice Questions

## Introduction 

<p align="center">
  <img src="imgs/PoE_ID.png" alt="PoE_ID Overview" width="600"/>
</p>

## 1. Dependencies
Use the following command to setup a conda environment and download required pacakages.
```
conda create -n PoE_ID python=3.10
conda activate PoE_ID
pip install -r requirements.txt
```

## 2. Run
```
bash script/option_id.sh
bash script/option_score.sh
bash script/mask.sh
bash script/EE_new.sh
bash script/few_shot.sh
```

## 3. Citations

If you find this repository helpful, please cite our paper.

```
@article{zhu2025optionidbasedeliminationmultiple,
      title={Option-ID Based Elimination For Multiple Choice Questions}, 
      author={Zhenhao Zhu and Bulou Liu and Qingyao Ai and Yiqun Liu},
      year={2025},
      eprint={2501.15175},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.15175}, 
}
```
