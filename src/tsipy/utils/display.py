from typing import NoReturn, Dict

terminal_color_tokens: Dict[str, str] = {
    "white": "\033[97m",
    "gray": "\033[90m",
    "purple": "\033[95m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
}

terminal_end_token: str = "\033[0m"


def cformat(string: str, color: str = None) -> str:
    if color is not None and color in terminal_color_tokens:
        start_token = terminal_color_tokens[color]
        end_token = terminal_end_token
        return start_token + string + end_token

    return string


def pformat(*args, shift: int = 50, level: int = 0, color: str = None) -> str:
    if level > 0:
        format_str = " " * (4 * level) + "{:<" + str(shift - 4 * level) + "}"
    else:
        format_str = "{:<" + str(shift) + "}"

    if len(args) == 1:
        format_str = cformat(format_str, color=color)
        return format_str.format(*args)

    format_str = format_str + "    ".join(["{}" for _ in range(len(args) - 1)])
    format_str = cformat(format_str, color=color)
    return format_str.format(*args)


def pprint(*args, shift: int = 50, level: int = 0, color: str = None) -> NoReturn:
    print(pformat(*args, shift=shift, level=level, color=color))


def pprint_block(
    *args, width: int = None, level: int = 0, color: str = None
) -> NoReturn:
    if level == 0:
        width = width if width is not None else 100
        block_str = "\n".join(["-" * width, pformat(*args), "-" * width])
    else:
        width = width if width is not None else 75
        block_str = "\n".join([pformat(*args, level=level), "-" * width])

    block_str = cformat(block_str, color=color)
    print("\n" + block_str)
