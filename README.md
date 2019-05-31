# KerasWrappers
Wrappers around the Sequential and Functional API of Keras

## Creating a model
```python
class MyModel(SequentialWrapper):
    def __init__(self):
        model = super().__init__('my_model')
	model.add(Dense(32, input_dim=784))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    def preprocess_x(self, data):
	# Optional preprocessing
        return super().preprocess_x(data)

    def preprocess_y(self, data):
	# Optional preprocessing
        return super().preprocess_y(data)

    def postprocess(self, data):
	# Optional postprocessing
        return super().postprocess(data)

my_model = MyModel()
```

```python
class MyModel(ModelWrapper):
    def __init__(self):
        model = super().__init__()
	model.add(Dense(32, input_dim=784))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    def preprocess_x(self, data):
	# Optional preprocessing
        return super().preprocess_x(data)

    def preprocess_y(self, data):
	# Optional preprocessing
        return super().preprocess_y(data)

    def postprocess(self, data):
	# Optional postprocessing
        return super().postprocess(data)

my_model = MyModel()
```

## Training
Training saves epochs (if not interrupted).
```python
my_model.train(x, y)
```
```python
from keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            ... ,
            horizontal_flip=True
          )
my_model.train_generator(generator, cat_train_x, cat_train_y, batch_size=64, epochs=5)
```

## Saving
```python
my_model.save_model()
```

## Loading
```python
retrained_classifier.load_model('my_model_35_epochs')
```

## Plotting
```python
my_model.plot_history(logy=False)
```

## TODO
work in progress...
