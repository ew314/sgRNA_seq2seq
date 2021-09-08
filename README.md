feedback seq2seq-attention model
================================
System Requirements and Installation
---------------------------------
Python == 2.7.16<br>
numpy == 1.16.5<br>
tensorflow == 1.7.0<br>
keras == 2.1.6<br>
sklearn == 0.20.3<br>
matplotlib == 2.2.3<br>

---
Relative activity predictor
---------------------------------
This model is trained to predict the relative activity of mismatched sgRNA with its original matched sgRNA.<br>


---
seq2seq-attention
---------------------------------
this model is trained to predict matched sgRNA or mismatched sgRNA with higher activity than matched sgRNA for input DNA target. Relative activity of mismatched sgRNA are predicted with the relative activity predictor above. Initial training set are letters_source_high_uniqe_22.txt and letters_target_high_uniqe_22.txt. Model will gain more input DNA - output sgRNA pairs from feedback_data with a feedback mechanism. Model's performance is evaluated with vaildation_data<br>
pre-trained model is in https://github.com/ew314/ew314/tree/main/sgRNA_designer/seq2seq_attention/trained_model<br>

---
Absolute activity predictor
---------------------------------
this model is trained to predict the absolute activity of matched sgRNA.<br>
pre-trained model is https://github.com/ew314/ew314/blob/main/sgRNA_designer/absolute_on_target_activity/NC_WT_float_model.h5<br>

---
Promotion potential classifier
---------------------------------
these two models are trained to classify which kind of DNA input would be predicted with matched sgRNA or mismatched sgRNA by seq2seq-attention model.<br>
One model is trained with DNA input in feedback_data and evaluated with DNA in vaildation_data. Another model is trained with DNA input in vaildation_data and evaluated with DNA in feedback_data.<br>
