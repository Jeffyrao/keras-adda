# keras-adda
This is an implementation of Adversarial Discriminative Domain Adaptation [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464) using Keras for the purpose of visualizing the intermediate activations in a variational approximation of total correlation.

The class implementing ADDA is in `adda.py`. 

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

