import os
import tkinter
import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

from prsoxs_xrr import XRR

ctk.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.refl = XRR(f"{os.getcwd()}/tests/TestData/Sorted/282.5")
        self.number_frames = len(self.refl.images)

        # configure window
        self.title("XRR Analysis")
        self.geometry(f"{1100}x{580}")

        # configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create main window
        self.frame = ctk.CTkFrame(self, width=200, height=200)
        self.frame.grid(
            row=1,
            column=2,
            columnspan=3,
            rowspan=3,
            padx=(20, 20),
            pady=(20, 20),
            sticky="nsew",
        )

        self.button = ctk.CTkButton(
            master=self,
            text="Plot Image",
            command=self.show_image(0),
        )
        self.button.grid(
            row=0,
            column=0,
            padx=(20, 20),
            pady=(20, 20),
            sticky="nsew",
        )

        self.slider = ctk.CTkSlider(
            master=self,
            from_=0,
            to=int(self.number_frames),
            number_of_steps=int(self.number_frames - 1),
            command=self.show_image,
        )
        self.slider.grid(
            row=1,
            column=0,
            padx=(20, 20),
            pady=(20, 20),
            sticky="nsew",
        )

    def show_image(self, i):
        fig, ax = plt.subplots()
        ax.imshow(self.refl.images[i])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().grid(
            row=1,
            column=2,
            padx=(20, 20),
            pady=(20, 20),
            sticky="nsew",
        )
        ###############    TOOLBAR    ###############
        toolbarFrame = ctk.CTkFrame(master=self)
        toolbarFrame.grid(row=0, column=2)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
        self.update()


if __name__ == "__main__":
    app = App()
    app.mainloop()
