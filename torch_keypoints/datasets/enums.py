from enum import Enum


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Person(Enum):
    R_ANKLE = 0
    R_KNEE = 0
    R_HIP = 0
    L_HIP = 0
    L_KNEE = 0
