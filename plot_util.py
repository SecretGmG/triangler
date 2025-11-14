import numpy as np
import matplotlib.pyplot as plt
from integrand_builder import CompiledIntegrand


def plot_complex_plane(xs, ys):
    """Plot a complex→complex function using HSV color encoding for phase and magnitude.
    xs is a 2D grid (from np.meshgrid) of complex-plane x-values, ys is the complex output.
    NaN or inf values in ys are handled gracefully and shown as transparent.
    """

    # Mask invalid data
    valid_mask = np.isfinite(ys)
    if not np.any(valid_mask):
        raise ValueError("All ys values are NaN or inf — nothing to plot.")

    # Compute phase and magnitude safely
    phase = np.angle(np.where(valid_mask, ys, 0))
    mag = np.abs(np.where(valid_mask, ys, 0))
    max_mag = np.nanmax(mag)
    mag = mag / max_mag if max_mag != 0 else mag

    # HSV mapping
    hue = (phase + np.pi) / (2 * np.pi)
    value = mag

    # HSV → RGB
    rgb = plt.cm.hsv(hue)
    rgb[..., :3] *= value[..., None]

    # Add transparency for invalid values
    alpha = np.where(valid_mask, 1.0, 0.0)
    rgb[..., -1] = alpha

    # Compute plotting extents (robust to NaNs)
    x_real = np.real(xs)
    y_imag = np.imag(xs)
    x_min, x_max = np.nanmin(x_real), np.nanmax(x_real)
    y_min, y_max = np.nanmin(y_imag), np.nanmax(y_imag)

    # Plot
    plt.imshow(
        rgb,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        aspect="equal",  # maintain correct aspect ratio
    )


def plot_complex(xs, ys):
    """
    Plot a real -> complex function
    """
    plt.plot(xs, ys.real, label="re")
    plt.plot(xs, ys.imag, label="im")


def plot_threshold_subtraction(compiled: CompiledIntegrand, x_lim, y_lim, x_axis = None, y_axis = None,  res = 300):
    if x_axis is None:
        x_axis = np.array([1,0,0])
    if y_axis is None:
        y_axis = np.array([0,1,0])
    
    x = np.linspace(x_lim[0], x_lim[1], res)
    y = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x, y)
    
    xs_plane = X + Y*1j
    ks_plane = (X[..., None] * x_axis + Y[..., None] * y_axis).reshape(-1, 3)
    ks_line = (x_axis[:, None] * x).T

    ks_plane_jac = np.sum(ks_plane**2, axis = 1)
    ks_line_jac = np.sum(ks_line**2, axis = 1)
    
    plt.figure(figsize=(20,10))
    plt.subplot(2,3,1)
    integrand = (compiled.eval_integrand(ks_plane)[:,0]*ks_plane_jac).reshape(res,res)
    plot_complex_plane(xs_plane, integrand)
    plt.subplot(2,3,2)
    counter_term = (compiled.eval_counterterm(ks_plane)[:,0]*ks_plane_jac).reshape(res,res)
    plot_complex_plane(xs_plane, counter_term)
    plt.subplot(2,3,3)
    subtracted = (compiled.eval_subtracted(ks_plane)[:,0]*ks_plane_jac).reshape(res,res)
    plot_complex_plane(xs_plane, subtracted)
    plt.subplot(2,3,4)
    plot_complex(x, compiled.eval_integrand(ks_line)[:,0] * ks_line_jac)
    plt.subplot(2,3,5)
    plot_complex(x, compiled.eval_counterterm(ks_line)[:,0] * ks_line_jac)
    plt.subplot(2,3,6)
    plot_complex(x, compiled.eval_subtracted(ks_line)[:,0] * ks_line_jac)
    plt.show()
