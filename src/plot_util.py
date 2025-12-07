import numpy as np
import matplotlib.pyplot as plt


def plot_complex_plane(xs, ys, ax = None, cmap_factor = 1):
    """Plot a complex→complex function using HSV color encoding for phase and magnitude.
    xs is a 2D grid (from np.meshgrid) of complex-plane x-values, ys is the complex output.
    NaN or inf values in ys are handled gracefully and shown as transparent.
    """

    if ax is None:
        ax = plt.gca()

    # Mask invalid data
    valid_mask = np.isfinite(ys)
    if not np.any(valid_mask):
        raise ValueError("All ys values are NaN or inf — nothing to plot.")

    # Compute phase and magnitude safely
    phase = np.angle(np.where(valid_mask, ys, 0))

    # HSV mapping
    hue = (phase + np.pi) / (2 * np.pi)
    value = np.abs(np.where(valid_mask, ys, 0))
    
    value = value / value.max()
    value = np.log(cmap_factor*value+1)
    value = value / value.max()

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


def get_contour(x_lim, y_lim, z_lim, f, res = 100):
    import pyvista as pv
    x = np.linspace(x_lim[0], x_lim[1], res)
    y = np.linspace(y_lim[0], y_lim[1], res)
    z = np.linspace(z_lim[0], z_lim[1], res)
    xs = np.stack(np.meshgrid(x, y, z), axis = -1)
    vals = f(xs)
    grid = pv.ImageData()
    grid.dimensions = np.array(vals.shape)
    grid.origin = (x[0], y[0], z[0])
    grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid.point_data["vals"] = vals.flatten(order="F")
    return grid.contour([0])

