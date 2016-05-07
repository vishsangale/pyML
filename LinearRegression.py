""" Linear Regression"""
import sys


class LinearRegression:
    def __init__(self, data_set):
        self.data_set = data_set
        pass

    def read_data_set(self):
        pass


if __name__ == "__main__":
    data_set_name = sys.argv[1]
    print data_set_name
    regression = LinearRegression(data_set_name)
    regression.read_data_set()
