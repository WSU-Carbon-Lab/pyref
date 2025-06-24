"""Interactive Image Masking in Python using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector


class InteractiveImageMasker:
    """
    A class to interactively create a mask on an image using a rectangular selector.

    Attributes
    ----------
    image : ndarray
        The original image to be masked.
    mask : ndarray
        The mask with the same size as the image, initialized to all ones.
    fig : matplotlib.figure.Figure
        The figure object for displaying the image.
    ax : matplotlib.axes.Axes
        The axes object for displaying the image.
    selector : matplotlib.widgets.RectangleSelector
        The rectangular selector for selecting regions on the image.

    Methods
    -------
    __init__(
        self,
        image,
        mask=None,
        title="Draw mask, press 't' to toggle, 'm' to save and close",
    ):
        Initializes the InteractiveImageMasker with the given image.

    on_select(eclick, erelease):
        Callback when the user makes a selection. Updates the mask and displays the
        masked image.

    toggle_selector(event):
        Enables/disables the selector based on key press (toggle with 't').

    return_mask(event):
        Prints a message and the mask when 'm' key is pressed.

    get_mask():
        Returns the mask.
    """

    def __init__(
        self,
        image,
        mask=None,
        title="Draw mask, press 't' to toggle, 'm' to save and close",
    ):
        self.image = image  # Original image
        if mask is None:
            self.mask = np.ones_like(image)  # Mask (same size as image)
        else:
            self.mask = mask.copy()
        self.fig, self.ax = plt.subplots()
        if self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title("Interactive Masking")
        self.ax.set_title(title)

        masked_image = self.image * self.mask
        self.ax.imshow(masked_image, cmap="terrain")

        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],  # Left-click  # type: ignore
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        plt.connect("key_press_event", self.toggle_selector)
        plt.connect(
            "key_press_event", self.return_mask
        )  # Connect mask retrieval on key press
        plt.show()

    def on_select(self, eclick, erelease):
        """
        User selection function.

        eclick and erelease are the press and release events.
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Set mask to 0 (blackout) outside the selected region
        self.mask[:, :] = 0  # Reset mask to all zero (blackout)
        self.mask[y1:y2, x1:x2] = 1  # Apply selection mask inside the region

        # Apply mask to the image (masking the outside part)
        masked_image = self.image * self.mask
        self.ax.clear()  # Clear the previous plot
        self.ax.imshow(masked_image, cmap="terrain")  # Show the masked image
        plt.draw()  # Redraw the figure with updated image

    def toggle_selector(self, event):
        """Enable/disable the selector based on key press (toggle with 't')."""
        if event.key in ["t", "T"]:
            if self.selector.active:
                self.selector.set_active(False)
            else:
                self.selector.set_active(True)

    def return_mask(self, event):
        """Return the mask when 'm' key is pressed."""
        if event.key in ["m", "M"]:
            print("Mask has been created, you can access it via 'get_mask' method.")
            plt.close(self.fig)

    def get_mask(self):
        """Return the mask."""
        return self.mask
