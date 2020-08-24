# NER-BiLSTM-CRF

# Introduction
Named Entity Recognition (NER) performed using a bidirectional-LSTM sequence labelling model with a CRF layer (BiLSTM-CRF), as implemented in Keras. The model is chosen for its ability to account for past and future tags (BiLSTM component), as well as its capacity to utilise sentence-level tag information (CRF component).

# Evaluation
The results are as follows: 
<br>
<br>![classification report](https://i.imgur.com/LckV1xu.png?1)

The results/accuracy can be visualised by comparing the actual and predicted NER tags in a random test case:
<br>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![test](https://i.imgur.com/PM4yMAg.png)
