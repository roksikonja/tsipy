import matplotlib as mpl
from matplotlib import style

from utils.constants import Constants as Const


class Visualizer:
    def __init__(self):
        style.use(Const.STYLE)
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
                "pgf.preamble": "\n".join(
                    [
                        "\\usepackage[utf8]{inputenc}",
                        "\\DeclareUnicodeCharacter{2212}{-}",
                    ]
                ),
                "font.size": Const.FONT_SIZE,
                "legend.fontsize": Const.LEGEND_FONT_SIZE,
                "legend.title_fontsize": Const.LEGEND_FONT_SIZE,
                "axes.labelsize": Const.AXIS_FONT_SIZE,
                "xtick.labelsize": Const.TICKS_FONT_SIZE,
                "ytick.labelsize": Const.TICKS_FONT_SIZE,
                "scatter.marker": Const.MARKER,
            }
        )

        mpl.rcParams["savefig.format"] = Const.OUT_FORMAT


def pprint(*args, shift=40):
    if len(args) < 2:
        raise ValueError("At least two arguments for printing.")

    format_str = (
        "{:<" + str(shift) + "}" + "\t".join(["{}" for _ in range(len(args) - 1)])
    )
    print(format_str.format(*args))
