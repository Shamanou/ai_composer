# Classical Piano Composer

This project allows you to train a neural network to generate midi music files that make use of multiple instruments

## Requirements

* Python 3.x
* Installing the following packages using pip:
    * Tensorflow
    * numpy
    * scipy

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in ./songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file
