"""
Pretty print utilities with indents, print blocks and colors.
"""
from typing import Dict

__all__ = ["cformat", "pformat", "pprint", "pprint_block"]


_terminal_color_tokens: Dict[str, str] = {
    "white": "\033[97m",
    "gray": "\033[90m",
    "purple": "\033[95m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
}

_terminal_end_token: str = "\033[0m"


def cformat(string: str, color: str = None) -> str:
    """Colors the input string."""
    if color is not None and color in _terminal_color_tokens:
        start_token = _terminal_color_tokens[color]
        end_token = _terminal_end_token
        return start_token + string + end_token

    return string


def pformat(*args: object, shift: int = 50, level: int = 0, color: str = None) -> str:
    """Pretty string formatting utility function into two columns.

    It formats arguments passed in two columns:
        - keyword (left aligned),
        - values (right aligned and separated by spaces).
    """
    if level > 0:
        format_str = " " * (4 * level) + "{:<" + str(shift - 4 * level) + "}"
    else:
        format_str = "{:<" + str(shift) + "}"

    if len(args) == 1:
        format_str = cformat(format_str, color=color)
        return format_str.format(*args)

    format_str += "    ".join(["{}" for _ in range(len(args) - 1)])
    format_str = cformat(format_str, color=color)
    return format_str.format(*args)


def pprint(*args: object, shift: int = 50, level: int = 0, color: str = None) -> None:
    """Pretty print utility function of arguments into two columns.

    Formatting is described :func:`pformat`.
    """
    print(pformat(*args, shift=shift, level=level, color=color))


def pprint_block(
    *args: object, width: int = None, level: int = 0, color: str = None
) -> None:
    """Pretty print utility function for code sections."""
    if level == 0:
        width = width if width is not None else 100
        block_str = "\n".join(["-" * width, pformat(*args), "-" * width])
    else:
        width = width if width is not None else 75
        block_str = "\n".join([pformat(*args, level=level), "-" * width])

    block_str = cformat(block_str, color=color)
    print("\n" + block_str)
