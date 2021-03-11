def pformat(*args, shift=40):
    if len(args) < 1:
        raise ValueError("At least two arguments for printing.")
    elif len(args) == 1:
        format_str = "{:<" + str(shift) + "}"
        return format_str.format(*args)

    format_str = (
        "{:<" + str(shift) + "}" + "    ".join(["{}" for _ in range(len(args) - 1)])
    )
    return format_str.format(*args)


def pprint(*args, shift=40):
    print(pformat(*args, shift=shift))


def pprint_block(*args, width=None, level=1):
    if level == 1:
        width = width if width is not None else 100
        block_str = "\n".join(["-" * width, *args, "-" * width])
    else:
        width = width if width is not None else 50
        args_str = [(4 * " ") + str(arg) for arg in args]
        block_str = "\n".join([*args_str, "-" * width])

    print("\n" + block_str)
