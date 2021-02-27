class GPConstants:
    NORMALIZATION = True
    CLIPPING = True

    TRAIN_INDUCING_PTS = False
    NUM_INDUCING_PTS = 1000

    PRIOR_POSTERIOR_SAMPLES = 3
    PRIOR_POSTERIOR_LENGTH = 2000


class SVGPConstants(GPConstants):
    BATCH_SIZE = 200
    MAX_ITER = 8000
    LEARNING_RATE = 0.005


class LocalSVGPConstants(GPConstants):
    X_WINDOW = 100
    WINDOW_FRACTION = 5
