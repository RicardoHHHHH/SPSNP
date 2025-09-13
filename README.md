# Reaction Prediction with NP-CLIP and Post-Modification

This repository implements the **SPSNP framework** for natural product structure prediction.  
Our approach integrates **NP-CLIP multimodal contrastive learning** (combining ChemBERTa-2 for SMILES and a Weisfeiler–Lehman Network (WLN) for molecular graphs), followed by **reaction center prediction**, **candidate generation and scoring** with a Weisfeiler–Lehman Difference Network (WLDN), and **rule-based post-modification**.  
This end-to-end workflow enables accurate prediction of reaction products and systematic generation of chemically plausible derivatives.  


## Reaction Core Identification

Reaction core identification codes are in **core_wln_global**.  
This folder also contains the weights obtained from **contrastive learning between molecular text representations**.  

To train the model:
```bash
python train.py --train ../data/XX.txt --hidden $HIDDEN --depth $DEPTH --save_dir $MODELDIR
```

To test the model:
```bash
python test.py --test ../data/XX.txt --hidden $HIDDEN --depth $DEPTH --model $MODELDIR > test.cbond
```

This prints the top 10 atom pairs in the reaction center for each reaction.  
Here `$MODELDIR` refers to a folder like `core_wln_global/core-300-3`.  

---

## Candidate Ranking

Codes are in **rank_diff_wln**.  
Note that you need to finish training/testing reaction core identification before candidate ranking.  

To train the model:
```bash
python nntrain_direct_useScores.py --train ../data/XX.txt --cand ../core_wln_global/train.cbond --hidden $HIDDEN --depth $DEPTH --ncand $NCAND --ncore $NCORE --save_dir $MODELDIR
```

Here `--cand` argument takes the file of predicted reaction centers from the previous pipeline.  
- `--ncore` sets the limit of reaction center size used for candidate generation.  
- `--ncand` sets the limit of number of candidate products for each reaction.  

To test the model:
```bash
python nntest_direct_useScores.py --test ../data/XX.txt --cand ../core_wln_global/test.cbond --hidden $HIDDEN --depth $DEPTH --ncand $NCAND --ncore $NCORE --save_dir $MODELDIR > test.cbond
```

This outputs the top 5 candidate products in one line for each reaction.  
Note that this script only outputs the bond type assignment over each atom pair (single/double/triple/delete, etc).  
You need to run the next script that generates SMILES strings of the products.  

---

## Molecular Decoration

Codes are in **decoration**.  
This folder contains scripts for **molecular post-modification**, enabling the exploration of structural variants beyond the original reaction outputs.  

---

## Data

All required data files are in the **data** folder:  
- `train.txt.proc`  
- `test.txt.proc`  
