class Settings:
    def __init__(self):
        self.__input_shape = (105, 105, 1)

    @property
    def input_shape(self):
        return self.__input_shape


if __name__ == "__main__":
    print(Settings().__input_shape)
