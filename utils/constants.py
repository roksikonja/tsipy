import matplotlib.colors as mcolors


class Constants(object):
    # Directories
    DATA_DIR = "./data"
    RESULTS_DIR = "./results"

    # RANDOMNESS
    RANDOM_SEED = 0

    # PLOTTING
    STYLE = "seaborn-colorblind"
    COLORS = list(mcolors.TABLEAU_COLORS)
    FIG_SIZE = (8, 6)
    OUT_FORMAT = "pdf"

    AXIS_FONT_SIZE = 18
    TICKS_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 16
    FONT_SIZE = 16
    X_TICKER = 4

    LW = None
    MARKER = "x"
    MARKER_SIZE = 3

    # UNITS
    DAY_UNIT = "TIME [mission days]"
    YEAR_UNIT = "TIME [year]"

    RATIO_UNIT = "RATIO [1]"
    TSI_UNIT = r"TSI $[W/m^2]$"
    DEGRADATION_UNIT = "DEGRADATION [1]"
