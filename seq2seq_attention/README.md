Mismatched sgRNA seq2seq-attention model
================================
This repository includes an approach to training Single-mismatched sgRNA seq2seq-attention model and Multi-mismatched sgRNA seq2seq-attention model. Pre-trained models also could directly be used to predicted mismatched sgRNA with input DNA target set.

![Flow](https://github.com/ew314/sgRNA_seq2seq/blob/main/seq2seq_attention/flow.jpg)

1.training Single/Multi-mismatched sgRNA seq2seq-attention model
--------------

    python Single_mismatched_sgRNA_seq2seqattention.15fold.py/Multi_mismatched_sgRNA_seq2seqattention.py

User need to edit scripts to modify some settings. <br>
`sourcefile` : 1978 DNA targets (/data/train_data/letters_source_uniqe_all.txt) as initial training set. We performed 1-5 fold cross validation with letters_source_uniqe_train_1/2/3/4/5.<br>
`targetfile` : 1978 matched sgRNA or high activity mismatched sgRNA (/data/train_data/letters_target_uniqe_all.txt) as initial training set. Their relative activity were recorded for further iteration.<br>
`feedback_filename`   : 40,000 DNA target sequences randomly chosen from Human GeCKOv2 Library[1] for feedback step to enlarge training set.<br>
`vaildation_filename` : 40,000 DNA target sequences randomly chosen from Human GeCKOv2 Library[1] for vaildation step to test model performance.<br>
`rel_predictor`       : Pre-trained relative activity predictor.<br>
`model_save_loc`      : Folder path for temporary trained models during 1200 epoches.<br>
`final_model_loc`     : Folder path for final trained model after 1200 epoches.<br>
`vocab_dict_path`     : Dict for encoding sequences into binary.<br>  
Pre-trained models were in Multi_final_model and Single_final_model.<br>
2.Predict mismatched sgRNA with trained seq2seq-attention model
--------------

    python Seq2seqattention.load_model.py

User need to edit scripts to modify some settings. <br>
`sourcefile` : 1978 DNA targets (/data/train_data/letters_source_uniqe_all.txt) as initial training set. We performed 1-5 fold cross validation with letters_source_uniqe_train_1/2/3/4/5.<br>
`targetfile` : 1978 matched sgRNA or high activity mismatched sgRNA (/data/train_data/letters_target_uniqe_all.txt) as initial training set. Their relative activity were recorded for further iteration.<br>
`pre_filename`   : DNA target sequences waited to be predicted mismatched sgRNAs.<br>
`vocab_dict_path`     : Dict for encoding sequences into binary.<br>  
`rel_predictor`       : Pre-trained relative activity predictor.<br>
`final_model`     : Pre-trained seq2seq-attention model.<br>
`output_file`     : Output file name.<br>

[[1]Sanjana NE, Shalem O, Zhang F: Improved vectors and genome-wide libraries for CRISPR screening. Nature methods 2014, 11(8):783-784.](https://www.nature.com/articles/nmeth.3047)<br>
