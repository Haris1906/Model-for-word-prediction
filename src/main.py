# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
import argparse
import codecs
import cv2
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from line_segmentation_functions import segment_words 

import random
import numpy as np

# Disable eager execution
tf.compat.v1.disable_eager_execution()

class FilePaths:
    "filenames and paths to data"
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '/content/12.png'
    fnCorpus = '../data/hindi_vocab.txt'

def train(model, loader):
    "train NN"
    epoch = model.lastEpoch  # Start from the last epoch
    bestCharErrorRate = float('inf')
    noImprovementSince = 0
    earlyStopping = 5
    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save(epoch)  # Pass the current epoch to save
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break

def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate

def infer(model, fnImg):
    "recognize text in image provided by file path"
    try:
        img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        if img is None:
            raise Exception("Image could not be read")
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, True)
        print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
        print('Probability:', probability[0])
        return recognized[0].replace(" ", "")
    except Exception as e:
        print(f"Error during inference: {e}")
        return "error"

def infer_by_web(path, option):
    decoderType = DecoderType.BestPath
    if option == "bestPath":
        decoderType = DecoderType.BestPath
        print("Best Path Execute")
    if option == "beamSearch":
        decoderType = DecoderType.BeamSearch
    print(open(FilePaths.fnAccuracy).read())
    model = Model(codecs.open(FilePaths.fnCharList, encoding="utf8").read(), decoderType)
    img = preprocess(cv2.imread(path, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0].replace(" ", "") + '"')
    print('Probability:', probability[0])
    return recognized[0].replace(" ", ""), probability[0]

import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def write_devnagari_text(img, text, position, font_path="/usr/share/fonts/truetype/noto/NotoSansDevanagari-Thin.ttf", font_size=30):
    """Writes Devanagari text directly onto an image using the specified font."""
    try:
        # Convert OpenCV image to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Load the specified Devanagari font
        font = ImageFont.truetype(font_path, font_size)
        
        # Draw text on image
        draw.text(position, text, font=font, fill=(0, 0, 0))  # Black text

        # Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error rendering Devanagari text: {e}")
    
    return img


def main():
    "main function"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the NN", action="store_true")
    parser.add_argument("--validate", help="validate the NN", action="store_true")
    parser.add_argument("--beamsearch", help="use beam search instead of best path decoding", action="store_true")
    parser.add_argument("--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    parser.add_argument("--image", help="path to input image for segmentation and OCR", type=str)
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    # Train or validate the model
    if args.train or args.validate:
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        open(FilePaths.fnCharList, 'w', encoding='UTF-8').write(str().join(loader.charList))
        open(FilePaths.fnCorpus, 'w', encoding='UTF-8').write(str(' ').join(loader.trainWords + loader.validationWords))

        if args.train:
            model = Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), decoderType, mustRestore=False, lastEpoch=32)
            train(model, loader)
        elif args.validate:
            model = Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), decoderType, mustRestore=True, lastEpoch=32)
            validate(model, loader)

    # Perform OCR with segmentation and reconstruct the page with predicted words
    model = Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), DecoderType.BestPath, mustRestore=True)

    # Perform OCR with segmentation and write text directly on image
    if args.image:
        output_dir = "cropped_images"
        os.makedirs(output_dir, exist_ok=True)
        bounding_boxes = segment_words(args.image, output_dir)
        original_image = cv2.imread(args.image)

        cropped_images = sorted(os.listdir(output_dir))
        for cropped_img_name in cropped_images:
            cropped_img_path = os.path.join(output_dir, cropped_img_name)
            cropped_img = cv2.imread(cropped_img_path, cv2.IMREAD_GRAYSCALE)
            if cropped_img is None:
                continue  # Skip invalid images
            
            predicted_text = infer(model, cropped_img_path)
            if cropped_img_name in bounding_boxes:
                x, y, w, h = bounding_boxes[cropped_img_name]
                original_image = write_devnagari_text(original_image, predicted_text, (x, y + h))

        output_path = "reconstructed_page.png"
        cv2.imwrite(output_path, original_image)
        print(f"Reconstructed page saved at: {output_path}")
        cv2.imshow("Reconstructed Page", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Default behavior: infer on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(codecs.open(FilePaths.fnCharList, encoding='utf-8').read(), decoderType, mustRestore=False)
        infer(model, FilePaths.fnInfer)

if __name__ == '__main__':
    main()
