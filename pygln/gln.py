def GLN(backend: str, *args, **kwargs):
    if backend == 'jax':
        from pygln.jax import GLN
        return GLN(*args, **kwargs)

    elif backend == 'numpy':
        from pygln.numpy import GLN
        return GLN(*args, **kwargs)

    elif backend == 'pytorch':
        from pygln.pytorch import GLN
        return GLN(*args, **kwargs)

    elif backend == 'tf':
        from pygln.tf import GLN
        return GLN(*args, **kwargs)

    else:
        raise NotImplementedError()
