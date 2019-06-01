import numpy as np
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import os
import pickle


class Wrapper:

    def __init__(self, name):
        self.name = name
        self.epochs = 0
        self.histories = dict()

    def get_name(self):
        return self.name

    def get_keras_model(self):
        raise NotImplementedError()

    def preprocess_x(self, data):
        return np.array(data).astype('float32')

    def preprocess_y(self, data):
        return np.array(data).astype('float32')

    def postprocess(self, data):
        return np.array(data)

    def train(self, x_train, y_train, x_test=None, y_test=None, **kwargs):
        print('Starting from', self.epochs, 'epochs.')
        x_train, y_train = self.preprocess_x(x_train), self.preprocess_y(y_train)
        if not(x_test is None or y_test is None):
            kwargs['validation_data'] = self.preprocess_x(x_test), self.preprocess_y(y_test)
        kwargs['batch_size'] = kwargs.get('batch_size', 32)
        kwargs['epochs'] = kwargs.get('epochs', 3)

        hist = self.get_keras_model().fit(x_train, y_train, **kwargs)

        for label, points in hist.history.items():
            self.histories[label] = self.histories.get(label, list()) + points
        self.epochs += kwargs['epochs']

    def train_generator(self, generator, x_train, y_train, x_test=None, y_test=None, batch_size=32, **kwargs):
        print('Starting from', self.epochs, 'epochs.')
        x_train, y_train = self.preprocess_x(x_train), self.preprocess_y(y_train)
        if not (x_test is None or y_test is None):
            kwargs['validation_data'] = self.preprocess_x(x_test), self.preprocess_y(y_test)
        kwargs['epochs'] = kwargs.get('epochs', 3)

        hist = self.get_keras_model().fit_generator(
            generator.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            **kwargs
        )

        for label, points in hist.history.items():
            self.histories[label] = self.histories.get(label, list()) + points
        self.epochs += kwargs['epochs']

    def plot_history(self, log_y=False):
        fig, axs = plt.subplots(2, 1, sharex='all', figsize=(12, 6))
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle('Training history', fontsize=16, y=0.94)

        for i, label in enumerate(['loss', 'acc']):
            ax = axs[i]
            x = list(range(1, len(self.histories[label]) + 1))
            if log_y:
                ax.semilog_y(x, self.histories[label], '-o', label=label)
                val_label = 'val_' + label
                ax.semilog_y(x, self.histories[val_label], '-o', label=val_label)
            else:
                ax.plot(x, self.histories[label], '-o', label=label)
                val_label = 'val_' + label
                ax.plot(x, self.histories[val_label], '-o', label=val_label)
            if log_y:
                label = 'log(' + label + ')'
            ax.set_ylabel(label, fontsize=12)
            ax.legend()

        plt.xlabel('epochs', fontsize=12)
        plt.show()

    def test(self, x_test, y_test):
        x_test, y_test = self.preprocess_x(x_test), self.preprocess_y(y_test)
        scores = self.get_keras_model().evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def predict_one(self, x_predict):
        x_predict = np.expand_dims(x_predict, axis=0)
        x_predict = self.preprocess_x(x_predict)
        return self.postprocess(self.get_keras_model().predict(x_predict))

    def predict_all(self, x_predict):
        x_predict = self.preprocess_x(x_predict)
        return self.postprocess(self.get_keras_model().predict(x_predict))

    def save_model(self, dir_name='models'):

        # Construct path
        model_name = self.get_name()
        model_name += '_' + str(self.epochs) + '_epochs'
        model_dir = os.path.join(os.getcwd(), dir_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_name)

        # Save history
        self.get_keras_model().save(model_path + '.h5')
        print('Saved trained model at ' + model_path + '.h5')

        # Save history
        with open(model_path + '.p', 'wb') as file:
            pickle.dump(self.histories, file)
        print('Saved training history at ' + model_path + '.p')

    def load_model(self, model_name, dir_name='models'):

        # Construct path
        model_name = model_name.rstrip('.h5')
        model_dir = os.path.join(os.getcwd(), dir_name)
        model_path = os.path.join(model_dir, model_name)

        # Load weights
        self.get_keras_model().load_weights(model_path + '.h5')
        print('Model weights loaded.')

        # Load history
        if os.path.isfile(model_path + '.p'):
            with open(model_path + '.p', 'rb') as file:
                self.histories = pickle.load(file)
            print('Training history detected.')
        else:
            print('No training history detected.')

        # Load epochs
        try:
            parts = model_name.split('_')
            self.epochs = int(parts[parts.index('epochs') - 1])
            print(self.epochs, 'epochs detected.')
        except (ValueError, IndexError):
            self.epochs = 0
            print('No epochs detected.')


class SequentialWrapper(Sequential, Wrapper):

    def __init__(self, name):
        Sequential.__init__(self)
        Wrapper.__init__(self, name)

    def get_keras_model(self):
        return self


class ModelWrapper(Model, Wrapper):

    def __init__(self, inputs, outputs, name):
        Model.__init__(self, inputs=inputs, outputs=outputs)
        Wrapper.__init__(self, name)

    def get_keras_model(self):
        return self

