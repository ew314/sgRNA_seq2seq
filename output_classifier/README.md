Promotion potential classifier
================================
This repository includes an approach to training two classifiers to detect which kind of DNA target input tend to be predicted with mismatched sgRNA output or matched sgRNA output by pre-trained seq2seq-attention model.<br>
We trained one model with predicted resuls of feedback set, validated in predicted resuls of validation set. And we trained another model with predicted resuls of validation set, validated in predicted resuls of feedback set.<br>

    python keras.pre.class.py

User need to edit scripts to modify some settings. <br>
`feedback_data_pre_result` : predicted result of feedback set from pre-trained multi-mismatch seq2seq-attention model.<br>
`prediction_data_pre_result` : predicted result of prediction set from pre-trained multi-mismatch seq2seq-attention model.<br>
`model_save_loc`   : Folder path for two trained classifier models.<br>
`figure_save_loc` : Folder path for validation results (ROC and PR-AUC figures).<br>

