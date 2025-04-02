import logging
from typing import List, Optional

import numpy as np
import ruptures as rpt

# Configure logging
logger = logging.getLogger(__name__)


def find_breakpoints(embeddings: List[np.ndarray]) -> List[int]:
    """
    Detects change points in a time series of high-dimensional embeddings.

    Uses the Kernel Change Point Detection (KernelCPD) algorithm from the
    `ruptures` library with fixed parameters. This method is well-suited for
    detecting distributional changes in embedding spaces. It automatically
    determines the number of breakpoints using a penalty value (PELT-like approach).

    Internal Parameters (Fixed):
        kernel: "rbf"
        gamma: 0.01
        min_size: 5
        penalty_scale: 1.0
        custom_cost: None

    Args:
        embeddings: A list of NumPy arrays, where each array is a
            high-dimensional embedding vector (e.g., 768 dimensions).

    Returns:
        A list of integer indices representing the locations of the detected
        change points. The indices mark the end of a segment. For example,
        if the result is [100, 250, 500], it means the segments are
        [0...99], [100...249], [250...499]. The last index is always the
        total number of samples.

    Raises:
        ValueError: If embeddings list is empty, or cannot be converted to a 2D
            NumPy array.
        TypeError: If elements in `embeddings` are not numeric.
        rpt.exceptions.BadSegmentationParameters: If `n_samples` is too small
            relative to `min_size`.
        ImportError: If the ruptures library is not installed (implicitly).
        Other exceptions from the underlying `numpy` or `ruptures` libraries
            during processing.
    """
    # --- Hardcoded Parameters ---
    kernel: str = "rbf"
    gamma: Optional[float] = 0.01
    min_size: int = 5
    penalty_scale: float = 1.0

    if not embeddings:
        logger.warning("Input embeddings list is empty. Returning no breakpoints.")
        return []

    # --- 1. Prepare Data ---
    # Convert list of embeddings to a NumPy array
    # Let potential exceptions (ValueError, TypeError) propagate
    X = np.array(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a list of 1D arrays, resulting in a 2D array. Got shape {X.shape}"
        )
    n_samples, n_features = X.shape
    logger.debug(f"Prepared data shape: {X.shape}")

    # Check for minimum size early, before algorithm configuration
    if n_samples < min_size * 2:
        logger.warning(
            f"Number of samples ({n_samples}) is less than twice the minimum segment size ({min_size}). "
            "Change point detection may not be meaningful or possible. Returning end index only."
        )
        return [n_samples]  # Return the end index as the only "breakpoint"

    # --- 2. Configure Algorithm ---
    algo: rpt.base.BaseEstimator
    effective_kernel_name: str

    logger.debug(f"Configuring KernelCPD with fixed kernel='{kernel}'...")
    params = {}
    if kernel == "rbf":
        current_gamma = gamma
        # The original code had a fallback if gamma was None, but here it's hardcoded
        if current_gamma is None:
            # This case should ideally not happen with hardcoded gamma, but keep for safety
            current_gamma = 1.0 / n_features
            logger.warning(
                f"Hardcoded gamma is None, defaulting to 1/n_features = {current_gamma:.4f}."
            )
        params["gamma"] = current_gamma
        logger.debug(f"  Using gamma={current_gamma}")

    # No need to check known_kernels as it's hardcoded to a valid one ("rbf")
    algo = rpt.KernelCPD(kernel=kernel, params=params, min_size=min_size)
    effective_kernel_name = kernel

    # --- 3. Fit and Predict using Penalty ---
    # The penalty value determines the trade-off between fitting the data
    # and the number of change points.
    penalty_value = penalty_scale * n_features * np.log(n_samples)
    logger.debug(
        f"Fitting model and predicting breakpoints using penalty={penalty_value:.2f} "
        f"(scale={penalty_scale}, n_features={n_features}, n_samples={n_samples})"
    )

    # Let fit and predict raise exceptions if they occur
    algo.fit(X)
    result_bkps = algo.predict(pen=penalty_value)

    logger.info(
        f"Predicted {len(result_bkps) - 1} breakpoints using kernel='{effective_kernel_name}' "
        f"and penalty={penalty_value:.2f}: {result_bkps}"
    )

    # --- 4. Return Result ---
    # The result includes the end index of the signal (n_samples).
    return result_bkps


# Example Usage (can be placed in a separate script or under if __name__ == "__main__":)
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     print("Generating dummy embedding data for example...")
#     n_samples_example = 200
#     n_dims_example = 50 # Lower dim for faster example generation
#     n_bkps_true_example = 3
#     signal_example, bkps_true_example = rpt.pw_constant(
#         n_samples_example, n_dims_example, n_bkps_true_example, noise_std=0.5, delta=(1, 3)
#     )
#     # Convert to list of arrays for the function signature
#     embeddings_list_example = [signal_example[i] for i in range(n_samples_example)]
#     print(f"Generated {len(embeddings_list_example)} embeddings of dimension {n_dims_example}")
#     print(f"True breakpoints were at indices: {bkps_true_example}")
#
#     # --- Call the function ---
#     print("\nCalling find_breakpoints with RBF kernel...")
#     detected_bkps_rbf = find_breakpoints(
#         embeddings_list_example,
#         kernel="rbf",
#         gamma=0.05, # May need tuning
#         min_size=3,
#         penalty_scale=1.5 # May need tuning
#     )
#     print(f"Detected breakpoints (RBF): {detected_bkps_rbf}")
#
#     print("\nCalling find_breakpoints with Cosine kernel...")
#     detected_bkps_cosine = find_breakpoints(
#         embeddings_list_example,
#         kernel="cosine",
#         min_size=3,
#         penalty_scale=0.8 # May need tuning differently than RBF
#     )
#     print(f"Detected breakpoints (Cosine): {detected_bkps_cosine}")
#
#
#     # --- Optional: Display results ---
#     print("\nPlotting results (displaying first dimension)...")
#     fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
#
#     # Plot RBF results
#     rpt.display(signal_example[:, 0], bkps_true_example, detected_bkps_rbf, ax=axes[0])
#     axes[0].set_title(f"KernelCPD (RBF) - Detected: {detected_bkps_rbf}")
#
#     # Plot Cosine results
#     rpt.display(signal_example[:, 0], bkps_true_example, detected_bkps_cosine, ax=axes[1])
#     axes[1].set_title(f"KernelCPD (Cosine) - Detected: {detected_bkps_cosine}")
#
#     plt.tight_layout()
#     plt.show()
