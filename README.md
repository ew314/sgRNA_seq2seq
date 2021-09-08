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
This model is trained to predict the relative activity of mismatched sgRNA with its original matched sgRNA, We basically followed the model architecture constructed by Marco https://static-content.springer.com/esm/art%3A10.1038%2Fs41587-019-0387-5/MediaObjects/41587_2019_387_MOESM4_ESM.html and used the relative activity data of mismatched sgRNAs from https://www.nature.com/articles/s41587-021-00946-z

---
seq2seq-attention
---------------------------------
This model is trained to predict matched sgRNA or mismatched sgRNA with high activity for input DNA target. Relative activity of mismatched sgRNA are predicted with the relative activity predictor above.<br>

---
Promotion potential classifier
---------------------------------
These models are trained to classify which kind of DNA input would be predicted with matched sgRNA or mismatched sgRNA by seq2seq-attention model.<br>

---
Absolute activity predictor
---------------------------------
this model is trained to predict the absolute off-target activity on genome.<br>

---
