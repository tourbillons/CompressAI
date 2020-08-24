def rename_key(key):
    """Rename keys to new format.
    """

    # Handle EntropyBottleneck move from nn.ParameterList to nn.Parameter's
    if key.startswith('entropy_bottleneck._biases.'):
        return f'entropy_bottleneck._bias{key[-1]}'

    if key.startswith('entropy_bottleneck._matrices.'):
        return f'entropy_bottleneck._matrix{key[-1]}'

    if key.startswith('entropy_bottleneck._factors.'):
        return f'entropy_bottleneck._factor{key[-1]}'

    return key


def load_pretrained(state_dict):
    """Convert state_dict keys to new format.
    """
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict
