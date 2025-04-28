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

Run in Web with Flask
```markdown
$ python upload.py
Validation character error rate of saved model: 12.988031%
Python: 3.6.4 |Anaconda, Inc.| (default, Mar 12 2018, 20:20:50) [MSC v.1900 64 bit (AMD64)]
Tensorflow: 1.8.0
Init with stored values from ../model/snapshot-2
Recognized: "होसले"
Probability: 0.7297366
```

```bash
@techreport{Devnagari-handwritten-word-recognition-2019,
  title={Devnagari Handwritten Word Recognition},
  author={Gautam Sushant},
  institution={Tribhuvan University},
  year={2019}
}
```
For the completion of the eighth semester in the Computer Science program at Tribhuvan University. July 2019.


