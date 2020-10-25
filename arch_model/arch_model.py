'''
'''
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Conv2D, Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda
from keras.regularizers import l2
try:
    from settings import Settings
except ModuleNotFoundError:
    from arch_model.settings import Settings


class ArchModel:
    def __init__(self):
        self.input_shape = Settings().input_shape
        self.msgs = {"error": {"layer": "> Error while adding {}˚ layer"}}
        self.model = None

    def get_arch(self, arch="siam"):
        if arch == "siam":
            try:
                self.model = self.__build_siamese()
            except NameError:
                print(f"> Error while trying to build architecture ({arch}).")
                self.model = None
        self.__compile()
        return self.model

    def __initialize_weights(self, shape, name=None, dtype=None):
        """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0
        and standard deviation of 0.01
        """
        return np.random.normal(loc=0.0, scale=1e-2, size=shape)

    def __initialize_bias(self, shape, name=None, dtype=None):
        """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard
        deviation of 0.01
        """
        return np.random.normal(loc=0.5, scale=1e-2, size=shape)

    def __build_siamese(self):
        """
        Model architecture based on the one provided in:
        http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """
        print(" ====== BUILDING CNN MODEL ======")
        layer_no = 0
        # Define a input layer (tensor) on the shape of the two input images
        left_input = Input(shape=self.input_shape)
        right_input = Input(shape=self.input_shape)

        # Model definition
        model = Sequential()

        # CNN
        layer_no += 1  # layer 1
        try:
            model.add(Conv2D(64,
                             (10, 10),
                             activation='relu',
                             input_shape=self.input_shape,
                             kernel_initializer=self.__initialize_weights,
                             kernel_regularizer=l2(2e-4))
                      )
            model.add(MaxPooling2D())
        except TypeError as error:
            print(self.msgs['error']['layer'].format(layer_no), "\n", error)
            exit()

        layer_no += 1  # layer 2
        try:
            model.add(Conv2D(128,
                             (7, 7),
                             activation='relu',
                             kernel_initializer=self.__initialize_weights,
                             bias_initializer=self.__initialize_bias,
                             kernel_regularizer=l2(2e-4))
                      )
            model.add(MaxPooling2D())
        except TypeError as error:
            print(self.msgs['error']['layer'].format(layer_no), "\n", error)
            exit()

        layer_no += 1  # layer 3
        try:
            model.add(Conv2D(128,
                             (4, 4),
                             activation='relu',
                             kernel_initializer=self.__initialize_weights,
                             bias_initializer=self.__initialize_bias,
                             kernel_regularizer=l2(2e-4))
                      )
            model.add(MaxPooling2D())
        except TypeError as error:
            print(self.msgs['error']['layer'].format(layer_no), "\n", error)
            exit()

        layer_no += 1  # layer 4
        try:
            model.add(Conv2D(256,
                             (4, 4),
                             activation='relu',
                             kernel_initializer=self.__initialize_weights,
                             bias_initializer=self.__initialize_bias,
                             kernel_regularizer=l2(2e-4))
                      )
            model.add(Flatten())
        except TypeError as error:
            print(self.msgs['error']['layer'].format(layer_no), "\n", error)
            exit()

        layer_no += 1  # layer 5
        try:
            model.add(Dense(4096,
                            activation='sigmoid',
                            kernel_initializer=self.__initialize_weights,
                            bias_initializer=self.__initialize_bias,
                            kernel_regularizer=l2(1e-3))
                      )
        except TypeError as error:
            print(self.msgs['error']['layer'].format(layer_no), "\n", error)
            exit()

        # Generate the features vectors (for each side)
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Adding custom layer to compute compute the absolute difference
        # between each side
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Adding the dense layer with the sigmoid unit to generate the
        # similarity score
        prediction = Dense(1, activation='sigmoid',
                           bias_initializer=self.__initialize_bias)(L1_distance)

        # Connect the inputs and the outputs
        siamese_net = Model(inputs=[left_input, right_input],
                            outputs=prediction)
        print("Done!")
        return siamese_net

    def __compile(self):
        "Compiling model..."
        optimizer = Adam(lr=0.00006)
        try:
            self.model.compile(loss="binary_crossentropy",
                               optimizer=optimizer)
            print("Compiling of model was successful!")
        except:
            print("> Error while trying to compile the model.")


if __name__ == "__main__":
    print("ArchModel test routine:")
    model_test = ArchModel().get_arch()
    try:
        model_test.summary()
    except AttributeError:
        print("> Model has not attribute summary!")

    try:
        model_test.compile()
    except:
        pass
