"""Interactive Jupyter widgets for Loader Class."""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Layout, VBox, interactive

from pyref.image import apply_mask, beamspot, reduction, dezinger_image
from pyref.loader import PrsoxrLoader


class SpotChecker:
    """Interactive Jupyter widgets for Loader Class."""

    def __init__(self, loader: PrsoxrLoader):
        self.energies = loader.energies  # List of energies
        self.shape = loader.shape  # Total number of frames
        self.roi = loader.roi  # Default ROI size
        self.blur_strength = loader.blur_strength  # Default blur strength
        self.selected_energy = None

    def render_spot(self, ax: plt.Axes):
        """Render the beamspot for a given frame and parameters."""

        def plot_frame(blur, roi, frame):
            meta = self.meta.row(frame, named=True)
            image = dezinger_image(np.reshape(meta["Raw"], meta["Raw Shape"]))
            masked = apply_mask(image, self.mask)
            bs = beamspot(masked, radius=blur)
            refl, refl_err = reduction(image, beam_spot=bs, box_size=roi)

            # Display the processed image
            ax.clear()
            ax.set_title(f"Specular Reflectance: {refl} ± {refl_err}")
            ax.imshow(image, cmap="terrain", interpolation="none")

            # Draw rectangles to highlight areas
            ax.add_patch(
                plt.Rectangle(
                    (0, bs[0] - roi),
                    image.shape[1],
                    2 * roi,
                    edgecolor="b",
                    facecolor="none",
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (bs[1] - roi, bs[0] - roi),
                    2 * roi,
                    2 * roi,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            ax.axis("off")
            # Update class variables
            self.blur_strength = blur
            self.roi = roi

        return plot_frame

    def display_widgets(self):
        """Display the interactive widgets for the SpotChecker."""
        # Dropdown for selecting energy
        energy_selector = widgets.Dropdown(
            options=self.energies,
            description="Energy",
            layout=Layout(width="200px"),
        )
        frame_selector = widgets.IntSlider(
            min=0,
            max=self.shape - 1,
            step=1,
            description="Frame",
            layout=Layout(width="300px"),
            continuous_update=False,
        )
        roi_selector = widgets.IntSlider(
            value=self.roi,
            min=0,
            max=50,
            step=1,
            description="ROI Size",
            layout=Layout(width="300px"),
        )
        blur_selector = widgets.IntSlider(
            value=self.blur_strength,
            min=0,
            max=50,
            step=1,
            description="Blur Strength",
            layout=Layout(width="300px"),
        )

        # Set up the figure and plotter
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plotter = self.render_spot(ax)

        # Interactive plot setup
        interactive_plot = interactive(
            plotter,
            blur=blur_selector,
            roi=roi_selector,
            frame=frame_selector,
        )

        # Display in a vertical box layout
        display(VBox([energy_selector, interactive_plot]))

        # Ensure selection of energy before using sliders
        def on_energy_change(change):
            if change["new"] is not None:
                self.selected_energy = change["new"]
                display(interactive_plot)

        energy_selector.observe(on_energy_change, names="value")
