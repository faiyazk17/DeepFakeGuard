import os
import warnings
import torch


def load_weights(target, path):
    """Load weights into target.

    `target` can be:
      - a full `Detector` instance (with `.head` and `.encoder`),
      - a specific module (e.g., `det.head`), or
      - a nn.Module compatible with a state dict.

    The saved file may be either a plain state_dict or a dict with keys
    like {'head': ..., 'encoder': ...} (as produced by the v3 training script).
    """
    if not os.path.exists(path):
        warnings.warn(f"Weights file not found at {path} — running in uninitialised mode.")
        return False

    try:
        state = torch.load(path, map_location=getattr(target, 'device', 'cpu'))
    except Exception as e:
        warnings.warn(f"Could not load weights file: {e}")
        return False

    # If state contains 'head' or 'encoder', try to load accordingly
    try:
        if isinstance(state, dict) and ('head' in state or 'encoder' in state):
            loaded_any = False
            if hasattr(target, 'head') and 'head' in state:
                try:
                    target.head.load_state_dict(state['head'], strict=False)
                    loaded_any = True
                except Exception as e:
                    warnings.warn(f"Failed to load head weights: {e}")
            if hasattr(target, 'encoder') and 'encoder' in state:
                try:
                    # load encoder tunables only (state['encoder'] may contain only norm layers)
                    target.encoder.model.load_state_dict(state['encoder'], strict=False)
                    loaded_any = True
                except Exception as e:
                    warnings.warn(f"Failed to load encoder weights: {e}")

            # If target itself is an nn.Module and top-level state provided
            if not loaded_any and hasattr(target, 'load_state_dict') and isinstance(state, dict):
                try:
                    target.load_state_dict(state, strict=False)
                    loaded_any = True
                except Exception:
                    pass

            return loaded_any

        # Otherwise assume it's a plain state dict for the given module
        if hasattr(target, 'load_state_dict'):
            try:
                target.load_state_dict(state, strict=False)
                return True
            except Exception as e:
                warnings.warn(f"Failed to apply state dict: {e}")
                return False

    except Exception as e:
        warnings.warn(f"Unexpected error loading weights: {e}")
        return False

    return False
