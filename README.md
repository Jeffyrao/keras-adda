# keras-adda
This is an implementation of Adversarial Discriminative Domain Adaptation [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464) using Keras for the purpose of visualizing the intermediate activations in a variational approximation of total correlation.

The class implementing ADDA is in `adda.py`. 

usage: adda.py [-h] [-s SOURCE_WEIGHTS] [-e START_EPOCH]
               [-n DISCRIMINATOR_EPOCHS] [-f]
               [-a SOURCE_DISCRIMINATOR_WEIGHTS]
               [-b TARGET_DISCRIMINATOR_WEIGHTS] [-t EVAL_SOURCE_CLASSIFIER]
               [-d EVAL_TARGET_CLASSIFIER]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE_WEIGHTS, --source_weights SOURCE_WEIGHTS
                        Path to weights file to load source model for training
                        classification/adaptation
  -e START_EPOCH, --start_epoch START_EPOCH
                        Epoch to begin training source model from
  -n DISCRIMINATOR_EPOCHS, --discriminator_epochs DISCRIMINATOR_EPOCHS
                        Max number of steps to train discriminator
  -f, --train_discriminator
                        Train discriminator model (if TRUE) vs Train source
                        classifier
  -a SOURCE_DISCRIMINATOR_WEIGHTS, --source_discriminator_weights SOURCE_DISCRIMINATOR_WEIGHTS
                        Path to weights file to load source discriminator
  -b TARGET_DISCRIMINATOR_WEIGHTS, --target_discriminator_weights TARGET_DISCRIMINATOR_WEIGHTS
                        Path to weights file to load target discriminator
  -t EVAL_SOURCE_CLASSIFIER, --eval_source_classifier EVAL_SOURCE_CLASSIFIER
                        Path to source classifier model to test/evaluate
  -d EVAL_TARGET_CLASSIFIER, --eval_target_classifier EVAL_TARGET_CLASSIFIER
                        Path to target discriminator model to test/evaluate


### Run Source Classifier on MNIST

To run the source encoder model on MNIST, use:

```
python adda.py [-e START_EPOCH]
```

### Run Target Discriminator on MNIST and SVHN:

To run the discriminator model on MNIST and SVHN to increase domain confusion, use:

```
python adda.py -f [-s SOURCE_WEIGHTS] [-n DISCRIMINATOR_EPOCHS] [-a SOURCE_DISCRIMINATOR_WEIGHTS] [-b TARGET_DISCRIMINATOR_WEIGHTS]
```

### Evaluate Source Classifier on MNIST:

```
python adda.py -t SOURCE_CLASSIFIER_WEIGHTS
```

### Evaluate Target Classifier on SVHN based on Domain Confusion:

```
python adda.py -t SOURCE_CLASSIFIER_WEIGHTS -d TARGET_DISCRIMINATOR_WEIGHTS
```
