# Devnagari Handwritten Word Recongition
### Description
Use Convolutional Recurrent Neural Network to recognize the Handwritten Word text image without pre segmentation into words or characters. Use CTC loss Function to train.

## <i> Basic Intuition on How it Works.

* First Use Convolutional Recurrent Neural Network to extract the important features from the handwritten line text Image.
* The output before CNN FC layer (512x100x8) is passed to the BLSTM which is for sequence dependency and time-sequence operations.
* Then CTC LOSS [Alex Graves](https://www.cs.toronto.edu/~graves/icml_2006.pdf) is used to train the RNN which eliminate the Alignment problem in Handwritten, since handwritten have different alignment of every writers. We just gave the what is written in the image (Ground Truth Text) and BLSTM output, then it calculates loss simply as `-log("gtText")`; aim to minimize negative maximum likelihood path.
* Finally CTC finds out the possible paths from the given labels. Loss is given by for (X,Y) pair is: ![Ctc_Loss](images/CtcLossFormula.png "CTC loss for the (X,Y) pair")
* Finally CTC Decode is used to decode the output during Prediction.
</i>

#### Dataset Used
IIT Devnagari Word Dataset. You can download it from [Devanagiri Dataset (IIIT-HW-Dev)](https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data).

###### The trained model CER=12.988% and trained on IIT Devnagari Word dataset with some additional created dataset.

To Train the model from scratch
```markdown
$ python main.py --train
```
To validate the model
```markdown
$ python main.py --validate
```
To Prediction
```markdown
$ python main.py
```

### NOTES
First delete all snapshots and set checkpoint to snapshots-0 in model and train the model for 30 epochs which will take time and then u will get snapshots stored in the model and use this snapshots to predict the words 

