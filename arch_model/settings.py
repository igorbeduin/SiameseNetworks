class Settings:
    def __init__(self):
        self.__width = 150
        self.__height = 150

    @property
    def input_shape(self):
        return (self.__height, self.__width, 1)


if __name__ == "__main__":
    print(Settings().__input_shape)
