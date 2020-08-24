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

    # Rename EntropyBottleneck keys
    modules_name = ['entropy_bottleneck', 'gaussian_conditional']
    new_names_map = {
        '_offset': 'cdf_offset',
        '_quantized_cdf': 'quantized_cdf',
        '_cdf_length': 'cdf_length',
    }
    for name in modules_name:
        for k, new_k in new_names_map.items():
            if f'{name}.{k}' == key:
                return f'{name}.{new_k}'

    # Deal with module trained with DataParallel
    if key.startswith('module.'):
        return key[7:]

    return key


def load_pretrained(state_dict):
    """Convert state_dict keys to new format.
    """
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict
