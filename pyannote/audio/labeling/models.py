#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import os.path

from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import merge
from keras.layers.wrappers import TimeDistributed

from pyannote.audio.callback import LoggingCallback
from keras.models import model_from_yaml


class SequenceLabeling(object):
    """Base class for sequence labeling

    Parameters
    ----------
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, log_dir=None):
        super(SequenceLabeling, self).__init__()
        self.log_dir = log_dir

    @classmethod
    def from_disk(cls, architecture, weights):
        """Load pre-trained sequence labeling from disk

        Parameters
        ----------
        architecture : str
            Path to architecture file (e.g. created by `to_disk` method)
        weights : str
            Path to pre-trained weight file (e.g. created by `to_disk` method)

        Returns
        -------
        sequence_labeling : SequenceLabeling
            Pre-trained sequence labeling model.
        """
        self = SequenceLabeling()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.labeling_ = model_from_yaml(yaml_string)
        self.labeling_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False, input_shape=None, model=None):
        """Save trained sequence labeling to disk

        Parameters
        ----------
        architecture : str, optional
            When provided, path where to save architecture.
        weights : str, optional
            When provided, path where to save weights
        overwrite : boolean, optional
            Overwrite (architecture or weights) file in case they exist.
        """

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        if model is not None:
            labeling = self.get_labeling(model)

        elif hasattr(self, 'labeling_'):
            labeling = self.labeling_

        elif input_shape is None:
            raise ValueError('Cannot save labeling to disk because input_shape is missing.')

        else:
            model = self.design_model(input_shape)
            labeling = self.get_labeling(model)

        if architecture:
            yaml_string = labeling.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            labeling.save_weights(weights, overwrite=overwrite)

    def fit(self, input_shape, generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=[], validation_data=None,
            nb_val_samples=None, class_weight={}, max_q_size=10):
        """Train model

        Parameters
        ----------
        input_shape :
        callbacks :

        For all other parameters, see Keras documentation for `fit_generator`
        """

        if not callbacks and self.log_dir:
            default_callback = LoggingCallback(self, log_dir=self.log_dir)
            callbacks = [default_callback]

        self.model_ = self.design_model(input_shape)
        self.labeling_ = self.get_labeling(self.model_)
        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks, validation_data=validation_data,
            nb_val_samples=nb_val_samples, class_weight=class_weight,
            max_q_size=max_q_size)

    def predict(self, sequence, batch_size=32, verbose=0):
        """
        """
        return self.labeling_.predict(
            sequence, batch_size=batch_size, verbose=verbose)


class LSTMSequenceLabeling(SequenceLabeling):
    """Bi-directional LSTM for sequence labeling

    Parameters
    ----------
    output_dim: int
        Number of output classes.
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [12] (i.e. one LSTM with 12 units)
    dense: list, optional
        List of output dimension of additionnal stacked dense layers.
        Defaults to [] (i.e. do not add any dense layer)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs.
        Defaults to mono-directional LSTMs.
    optimizer: str, optional
        Keras optimizer. Defaults to 'rmsprop'.
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, output_dim, lstm=[12], dense=[],
                 bidirectional=False, optimizer='rmsprop',
                 log_dir=None):

        super(LSTMSequenceLabeling, self).__init__(log_dir)

        self.output_dim = output_dim
        self.lstm = lstm
        self.dense = dense
        self.bidirectional = bidirectional
        self.optimizer = optimizer

    def design_labeling(self, input_shape):
        """Create Keras labeling model

        (The end user does not to use this method.)

        Parameters
        ----------
        input_shape : (n_samples,n_features) tuple
            Expected shape of input sequences
        """

        inputs = Input(shape=input_shape,
                       name="labeling_input")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            if i:
                # all but first LSTM
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               output_dim=output_dim,
                               return_sequences=True,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(forward)
                if self.bidirectional:
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    output_dim=output_dim,
                                    return_sequences=True,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(backward)
            else:
                # first LSTM
                forward = LSTM(name='forward_{i:d}'.format(i=i),
                               # input_shape=input_shape,
                               output_dim=output_dim,
                               return_sequences=True,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(x)
                if self.bidirectional:
                    backward = LSTM(name='backward_{i:d}'.format(i=i),
                                    go_backwards=True,
                                    #input_shape=input_shape,
                                    output_dim=output_dim,
                                    return_sequences=True,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(x)

        # concatenate forward and backward
        if self.bidirectional:
            # FIXME -- check value of concat_axis=1
            x = merge([forward, backward], mode='concat', concat_axis=2)
        else:
            x = forward

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            x = TimeDistributed(Dense(output_dim,
                                      activation='tanh',
                                      name='dense_{i:d}'.format(i=i)))(x)

        # one dimension per class
        outputs = TimeDistributed(Dense(self.output_dim, activation='softmax'))(x)

        return Model(input=inputs, output=outputs)

    def get_labeling(self, model):
        return model

    def design_model(self, input_shape):
        """Create Keras model and get ready for training

        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape, name="input")
        labels = self.design_labeling(input_shape)(inputs)
        model = Model(input=inputs, output=labels)

        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model