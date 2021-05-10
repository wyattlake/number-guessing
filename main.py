import numpy
import keras

if __name__ == "__main__":
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = numpy.array([6.0, 4.0, 8.0, 10.0, 2.0, 12.0], dtype=float)
    ys = numpy.array([27.0, 4.0, 256.0, 3125.0, 1.0, 46656.0], dtype=float)

    model.fit(xs, ys, epochs=100)
    print(model.predict([4.0]))
