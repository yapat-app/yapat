def get_embedding_model(method_name, *args, **kwargs):
    """
    Retrieve the appropriate embedding model based on the given method name.

    Parameters:
    - method_name (str): The name of the embedding method to use.
    - *args: Additional positional arguments to pass to the embedding model.
    - **kwargs: Additional keyword arguments to pass to the embedding model.

    Returns:
    - An instance of the specified embedding model.

    Raises:
    - ValueError: If the provided method_name is not recognized.
    """
    if method_name == "birdnet":
        from embeddings.birdnet import BirdnetEmbedding
        return BirdnetEmbedding(*args, **kwargs)

    elif method_name == "acoustic_indices":
        from embeddings.acoustic_indices import AcousticIndices
        return AcousticIndices(*args, **kwargs)

    elif method_name == "vae":
        from embeddings.vae import VAEEmbedding
        return VAEEmbedding(*args, **kwargs)

    else:
        raise ValueError(f"Unknown embedding method: {method_name}")
