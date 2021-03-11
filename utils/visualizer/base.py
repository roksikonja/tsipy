import matplotlib as mpl
from matplotlib import style

from ..constants import Constants as Const


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
