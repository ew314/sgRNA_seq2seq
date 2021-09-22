absolute_off_target_activity
================================
This repository includes an approach to train one model predicting sgRNA absolute off target activity. We first trained one on-target activity predictor with matched wild-type Cas9 sgRNAs activity measured by Wang[1]. Then we predicted the on-target activity of these 1978 matched sgRNAs from Marco[2] and transferred the relative activity of those 26249 mismatched sgRNAs into absolute off-target activity. These sgRNAs were then used to train absolute off target activity predictor.<br>

    python keras.off_target.activity.py
    
User need to edit scripts to modify some settings. <br>
`on_filename` : SgRNAs on-target activity from Wang.<br>
`rel_filename` : Mismatched sgRNAs relative activity from Marco.<br>
`model_save_loc`   : Folder path for trained absolute off-target activity predictor.<br>
`figure_save_loc` : Folder path for validation results (regression analysis between measured activity and predicted activity).<br>

[[1].Wang D, Zhang C, Wang B, Li B, Wang Q, Liu D, Wang H, Zhou Y, Shi L, Lan F et al: Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning. Nature Communications 2019, 10(1):4284](https://www.nature.com/articles/s41467-019-12281-8)<br>
[[2].Jost, M., Santos, D.A., Saunders, R.A. et al. Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs. Nat Biotechnol 38, 355–364 (2020).](https://www.nature.com/articles/s41587-021-00946-z)<br>
