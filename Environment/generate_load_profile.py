import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def _load_generator(model_path):
    """
    Try to load in order:
     1) Keras v3 single-file (.keras)
     2) If given a directory that is a SavedModel, wrap it with keras.layers.TFSMLayer
    """
    # 1) if user passed the .keras file directly
    if os.path.isfile(model_path) and model_path.endswith('.keras'):
        return tf.keras.models.load_model(model_path, compile=False)

    # 2) if user passed a directory, check for generator.keras inside it
    if os.path.isdir(model_path):
        keras_inside = os.path.join(model_path, 'generator.keras')
        if os.path.isfile(keras_inside):
            return tf.keras.models.load_model(keras_inside, compile=False)

        # 3) fallback: directory is a SavedModel -> wrap with TFSMLayer (inference-only)
        try:
            from keras.layers import TFSMLayer
            layer = TFSMLayer(model_path, call_endpoint='serving_default')
            return tf.keras.Sequential([layer])
        except Exception as e:
            raise RuntimeError(
                f"Directory looks like a SavedModel but couldn't wrap with TFSMLayer: {e}"
            ) from e

    # 4) try appending .keras if user passed '.../generator'
    candidate = model_path + '.keras'
    if os.path.isfile(candidate):
        return tf.keras.models.load_model(candidate, compile=False)

    raise FileNotFoundError(f"Could not find a loadable model at '{model_path}' (looked for .keras and SavedModel dir).")

def generate_profile(model_path, seed=None, as_flat=True, save_csv=None):
    """
    Load a saved generator model and return one generated profile.
    - model_path: path to the saved generator folder (e.g. 'generated_models/V2-1_relu/generator')
    - seed: optional integer seed for reproducibility
    - as_flat: return flattened 1D array when True, otherwise return array with original shape
    - save_csv: optional path to save the generated profile as CSV
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    model = _load_generator(model_path)
    # infer noise dim from model input shape: (None, noise_dim)
    noise_shape = model.input_shape
    if len(noise_shape) < 2 or noise_shape[1] is None:
        raise ValueError(f"Cannot infer noise dim from model input shape: {noise_shape}")
    noise_dim = int(noise_shape[1])

    noise = np.random.normal(size=(1, noise_dim)).astype(np.float32)
    generated = model.predict(noise)
    profile = generated[0]

    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or '.', exist_ok=True)
        np.savetxt(save_csv, profile.flatten(), delimiter=',')

    return profile.flatten() if as_flat else profile

def generate_profile_one_day(model_path, seed=None, save_csv=None):
    """
    Load saved generator and return one RANDOM day (96 points, 15-min steps) from the generated week (672 points).
    - model_path: path to the saved generator folder
    - seed: optional integer seed for reproducibility
    - save_csv: optional path to save the 1-day profile as CSV
    Returns: 1D numpy array length 96.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    model = _load_generator(model_path)

    # infer noise dim
    input_shape = model.input_shape
    if not input_shape or len(input_shape) < 2 or input_shape[1] is None:
        raise ValueError(f"Cannot infer noise dim from model input shape: {input_shape}")
    noise_dim = int(input_shape[1])

    noise = np.random.normal(size=(1, noise_dim)).astype(np.float32)
    generated = model.predict(noise)
    sample = generated[0]

    # flatten to 1D if necessary
    if sample.ndim > 1:
        sample = sample.reshape(-1)

    WEEK_LEN = 7 * 24 * 4   # 672
    DAY_LEN = 24 * 4        # 96

    if sample.size != WEEK_LEN:
        raise ValueError(f"Expected generated week length {WEEK_LEN}, got {sample.size}")

    day_idx = np.random.randint(0, 7)
    start = day_idx * DAY_LEN
    day_profile = sample[start:start + DAY_LEN].astype(np.float32)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or '.', exist_ok=True)
        np.savetxt(save_csv, day_profile, delimiter=',')

    return day_profile


# Example usage:
profile = generate_profile(r'generated_models/V2-1_relu/generator')
