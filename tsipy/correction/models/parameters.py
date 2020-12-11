class ExpConstants:
    MAX_FEVAL = 10000


class ExpLinConstants(ExpConstants):
    pass


class MRConstants:
    INCREASING = False
    Y_MAX = 1
    Y_MIN = 0
    OUT_OF_BOUNDS = "clip"


class SMRConstants(MRConstants):
    NUMBER_OF_POINTS = 999
    LAM = 1
