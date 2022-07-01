feedback seq2seq-attention model
================================
System Requirements and Installation
---------------------------------
Python == 2.7.16<br>
numpy == 1.16.5<br>
tensorflow == 1.7.0<br>
keras == 2.1.6<br>
scipy == 1.2.1<br>
sklearn == 0.20.3<br>
matplotlib == 2.2.3<br>

---
Relative activity predictor
---------------------------------
This model is trained to predict the relative activity of mismatched sgRNA with its original matched sgRNA.

---
seq2seq-attention
---------------------------------
This model is trained to predict matched sgRNA or mismatched sgRNA with high activity for input DNA target. Relative activity of mismatched sgRNA are predicted with the relative activity predictor above.<br>

---
output classifier
---------------------------------
These models are trained to classify which kind of DNA input would be predicted with matched sgRNA or mismatched sgRNA by seq2seq-attention model.<br>

---
Absolute activity predictor
---------------------------------
this model is trained to predict the absolute off-target activity on genome.<br>

---
Supplementary Tables.docx
---------------------------------
The Supplementary Table 1,2,3 in article Wei-Xin Hu, Yu Rong, Yan Guo, Feng Jiang, Wen Tian, Hao Chen, Shan-Shan Dong, Tie-Lin Yang, ExsgRNA: reduce off-target efficiency by on-target mismatched sgRNA, Briefings in Bioinformatics, 2022;, bbac183(https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac183/6587171).<br>

---
