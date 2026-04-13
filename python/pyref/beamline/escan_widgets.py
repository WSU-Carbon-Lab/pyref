"""Energy scan (ESCAN) widget for ALS beamline script generation.

This module provides widgets for generating energy scan run files for
ALS beamline 11.0.1.2.
"""
# Base Packages
import contextlib
import inspect
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

# Visualization Packages
import ipywidgets as widgets

# Math packages
import numpy as np
import pandas as pd
from IPython.display import display

from pyref.beamline.base_widgets import (
    ALS_MeasurementWidget,
)


class ESCAN_ScriptGen(ALS_MeasurementWidget):
    EXPERIMENT_NAME = "ESCAN"
    JSON_TITLE = "ESCAN"
    DEFAULT_NAME = "carbon_nexafs"
    DEFAULT_PARAMETERS = {
        "Step 1":[270.0, 280.0, 2, 0.5],
        "Step 2":[280.0, 284.0, 0.2, 1],
        "Step 3":[284.0, 286.0, 0.1, 0.5],
        "Step 4":[286.0, 292.0, 0.2, 0.5],
        "Step 5":[292.0, 300.0, 1, 0.5],
        "Step 6":[300.0, 370.0, 10, 1]
    }
    BUTTON_SIZE = widgets.Layout(width='10rem')
    LABEL_SIZE = widgets.Layout(width='5rem')
    TABLE_SIZE = widgets.Layout(widget='50rem', border='2px black', padding='5px')
    JSON_BUTTON_STYLE = {'button_color':'#672E45', 'text_color':'white'}
    OTHER_BUTTON_STYLE = {'button_color':'#007681', 'text_color':'white'}


    def __init__(self, path=None, **kwargs):
        self._save_dir = path if path is not None else ""
        parameters = self.DEFAULT_PARAMETERS.copy()
        super().__init__(constant_motor_title='<b>ESCAN Script Generator</b>') # Build the initial table
        # Title information
        # Name that will be used to save the
        self.save_name = widgets.Text(
            value="",
            description = "Name of Script: ",
            layout=widgets.Layout(width='300px'),
            style = {'description_width': '90px'}
        )

        # Browse directory functions
        self.browse_directory_button = widgets.Button(
            description="Choose Save Directory", layout=self.BUTTON_SIZE,
            style=self.OTHER_BUTTON_STYLE
        )
        self.browse_directory_button.on_click(self.browse_save_directory)
        # Display Save Directory
        self.display_save_directory = widgets.HTML(
            value = "<b>Save Directory:</b>" + str(self._save_dir),
            description = ""
        )
        self.directory_buttons = widgets.HBox([self.browse_directory_button, self.display_save_directory])

        # Initialize table // The double list is a mistake that I amn just going to live with forever
        self.table_titles = [[
                    widgets.HTML(value="<b>Step</b>", layout=self.LABEL_SIZE),
                    widgets.HTML(value="<b>Start</b>", layout=self.LABEL_SIZE),
                    widgets.HTML(value="<b>Stop</b>", layout=self.LABEL_SIZE),
                    widgets.HTML(value="<b>Delta</b>", layout=self.LABEL_SIZE),
                    widgets.HTML(value="<b>Exposure</b>", layout=self.LABEL_SIZE)
        ]]
        # Make the initial rows using this function
        self.make_table(parameters)

        self.table = widgets.VBox(self.update_table())

        # Create a button to add or remove columns and rows
        self.add_step_button = widgets.Button(description="Add step", layout=self.BUTTON_SIZE)
        self.add_step_button.on_click(self.add_step)

        self.remove_step_button = widgets.Button(description="Remove step", layout=self.BUTTON_SIZE)
        self.remove_step_button.on_click(self.remove_step)

        # Additional options for the Escan
        self.repeat_title = widgets.HTML(
            value = "Number of Repeats: ",
            description = ""
        )
        self.repeat_value = widgets.IntText(
            value = 1,
            layout=self.LABEL_SIZE
        )
        self.repeat_option = widgets.HBox(children=[self.repeat_title, self.repeat_value])

        # Put them into the control buttons item
        self.control_buttons = widgets.HBox(children=[self.add_step_button, self.remove_step_button])

        # Save button to JSON
        self.save_json_button = widgets.Button(
            description = "Save Script to JSON",
            layout=self.BUTTON_SIZE,
            style = self.JSON_BUTTON_STYLE
        )
        self.load_json_button = widgets.Button(
            description = "Load Script from JSON",
            layout=self.BUTTON_SIZE,
            style = self.JSON_BUTTON_STYLE
        )
        self.load_json_button.on_click(self.load_json)
        self.save_json_button.on_click(self.save_json)
        self.json_buttons = widgets.HBox(children=[self.save_json_button, self.load_json_button])

        self.save_for_nexafs = widgets.Button(
            description = "Save Single Motor Scan",
            layout = self.BUTTON_SIZE,
            style = self.OTHER_BUTTON_STYLE
        )
        self.save_for_nexafs.on_click(self.save_nexafs)
        self.save_for_rsoxs = widgets.Button(
            description = "Save CCD scan",
            layout = self.BUTTON_SIZE,
            style = self.OTHER_BUTTON_STYLE
        )
        self.save_for_rsoxs.on_click(self.save_rsoxs)

        self.save_scan_buttons = widgets.HBox(children=[self.save_for_nexafs, self.save_for_rsoxs])


        # Style the table
        self.table.layout=self.TABLE_SIZE

        self.display_table = self.build_display_table()
        display(self.display_table)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, val):
        self._save_dir=val
        self.display_save_directory.value = "<b>Save Directory: </b>" + str(self._save_dir)

    def make_table(self, d):
        for i, (key, item) in enumerate(d.items()):
            if isinstance(item, list):
                setattr(
                    self,
                    'variable_'+str(i+1),
                    [
                        [widgets.Label(value=key, layout=self.LABEL_SIZE)] +
                        [widgets.FloatText(value=ii, layout=self.LABEL_SIZE) for ii in item]
                    ]
                )

    def add_step(self, b):
        total_cols = len(self.table_titles[0])-1
        # Add a new row with FloatText widgets
        num_steps = len([attr for attr in dir(self) if attr.startswith("variable_")])
        # Create the new attribute
        setattr(
            self,
            "variable_"+str(num_steps+1),
            [
                #[widgets.Text(value="", layout=self.label_size)] +
                [widgets.Label(value="Step "+str(num_steps+1), layout=self.LABEL_SIZE)] +
                [widgets.FloatText(value=i, layout=self.LABEL_SIZE) for i in np.arange(total_cols)]
                #[widgets.Dropdown(options=AVAILABLE_MOTORS, value=::)]
            ]
        )
        getattr(self, "variable_"+str(num_steps+1))
        self.table.children = self.update_table()

    def remove_step(self, b):
        final_step = len([attr for attr in dir(self) if attr.startswith("variable_")])
        # No steps to remove
        if final_step == 1:
            print("No steps to delete")
            return 0
        for motor in self.get_table():
            if "variable_" in motor and str(final_step) in motor:
                delattr(self,motor)

        # Update the table
        self.table.children = self.update_table()

    def build_display_table(self):
        build_table = []
        build_table.extend([self.constant_motor_title]) # The title of fixed values
        build_table.extend([self.save_name]) # The title of fixed values
        build_table.extend([self.directory_buttons])
        build_table.extend([widgets.HTML(value="<I>Set Delta=0 to add a single energy</I>")])
        build_table.extend([self.control_buttons]) # optional buttons
        build_table.extend([self.table])  # table object if needed
        build_table.extend([self.repeat_option])
        build_table.extend([self.json_buttons]) # Save as JSON button
        build_table.extend([self.save_scan_buttons])
        return widgets.VBox(build_table)

    def browse_save_directory(self, b):
        self.save_dir = self.browse_directory()

    def browse_directory(self):
        import gc
        root = None
        selected_folder = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            initial_dir = str(self._save_dir) if self._save_dir else None
            selected_folder = filedialog.askdirectory(
                title="Select Folder",
                initialdir=initial_dir,
                parent=root
            )
        except Exception as e:
            print(f"Error in directory browser: {e}")
            selected_folder = None
        finally:
            if root is not None:
                try:
                    root.update()
                    root.update_idletasks()
                    for widget in root.winfo_children():
                        widget.destroy()
                    root.destroy()
                except:
                    pass
                with contextlib.suppress(Exception):
                    root.quit()
                root = None
                gc.collect()

        if selected_folder:
            return Path(selected_folder)
        return None

    def save_json(self, b=None):
        df = self.output_dict()
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.json'

        try:
            with open(SAVEPATH, 'w') as f:
                f.write(f"# {self.JSON_TITLE}\n")
                json.dump(df, f, indent=4)
        except json.JSONDecodeError as e:
            msg = f"Error encoding to JSON: {e}"
            raise json.JSONDecodeError(msg)

    def load_json(self, expected_title="", b=None):
        import gc
        root = None
        selected_file = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            initial_dir = str(self._save_dir) if self._save_dir else None
            selected_file = filedialog.askopenfilename(
                title="Select JSON File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=initial_dir,
                parent=root
            )
        except Exception as e:
            print(f"Error in file dialog: {e}")
            import traceback
            traceback.print_exc()
            selected_file = None
        finally:
            if root is not None:
                try:
                    root.update()
                    root.update_idletasks()
                    for widget in root.winfo_children():
                        widget.destroy()
                    root.destroy()
                except:
                    pass
                with contextlib.suppress(Exception):
                    root.quit()
                root = None
                gc.collect()

        if selected_file:
            try:
                with open(selected_file) as f:
                    title_line = f.readline().strip()
                    if title_line == f"# {self.JSON_TITLE}":
                        df = json.load(f)
                    else:
                        print(f"Not an appropriate script. Expected title: {self.JSON_TITLE}, got: {title_line}")
                        return None
                self.clean_slate()
                self.make_table(df)
                self.table.children = self.update_table()
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                import traceback
                traceback.print_exc()
                return None
        return None


    def save_nexafs(self, b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'

        df = self.output_dict()
        full_energy_scan = self._build_escan(df)[0] # Build the Escan
        # Repeat the Escan if it is requested
        full_energy_scan = [full_energy_scan]*self.repeat_value.value
        full_energy_scan = np.concatenate(full_energy_scan, axis=0)
        # Save it to disk
        np.savetxt(SAVEPATH, full_energy_scan, fmt='%.2f')
        # Cleanup the output because the ALS requires specific things
        try:
            with open(SAVEPATH) as f:
                lines = f.readlines()
            if not lines: # Empty file
                return
            # Edit first line to remove 'exposure' from headers
            if 'Exposure' in lines[0]:
                lines[0] = lines[0].replace('\tExposure' , '')
            # Edit last line to remove any carriage return that may exist
            if '\n' in lines[-1]:
                lines[-1] = lines[-1].replace('\n', '')
            with open(SAVEPATH, 'w') as f:
                f.writelines(lines)
        except FileNotFoundError:
            print(f"Error: File not found at {SAVEPATH}")
        except Exception as e:
            print(f"An error occured: {e}")
        del lines #Remove it from memory (it can be large)

    def save_rsoxs(self, b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'

        Motor_Order = ['Beamline Energy', 'CCD Camera Shutter Inhibit', 'Exposure']

        df = self.output_dict()
        escan_details = self._build_escan(df)
        Escan = escan_details[0] # The energies
        Exposure = escan_details[1] # The exposures

        # Build the sample dataframe
        temp_scan = {}
        temp_scan['Beamline Energy'] = Escan
        temp_scan['Exposure'] = Exposure
        df_samp = pd.DataFrame(temp_scan)

        # Add CCD Inhibit, first and last, all exposure times
        df_samp['CCD Camera Shutter Inhibit'] = np.full_like(Escan, 0) # Add the column
        for expo in np.unique(Exposure):
            df_samp = pd.concat([df_samp.loc[0].to_frame().T, df_samp], ignore_index=True) # Add a frame at the beginning of the sample
            df_samp = pd.concat([df_samp, df_samp.loc[df_samp.index[-1]].to_frame().T], axis=0, ignore_index=True) # Add a frame at the end of the sample

            df_samp.loc[0, 'CCD Camera Shutter Inhibit'] = 1 # Close the shutter
            df_samp.loc[df_samp.index[-1], 'CCD Camera Shutter Inhibit'] = 1

            df_samp.loc[0, 'Exposure'] = expo # Set exposure
            df_samp.loc[df_samp.index[-1], 'Exposure'] = expo

        # Cleanup the output because the ALS requires specific things
        cols = list(df_samp.columns)
        Ordered_Cols = [item for item in Motor_Order if item in cols]
        df_samp = self._iterate_dataframe(df_samp[Ordered_Cols], self.repeat_value.value) # Repeat the scan if it is requested

        df_samp.to_csv(SAVEPATH, index=False, sep='\t')

        try:
            with open(SAVEPATH) as f:
                lines = f.readlines()
            if not lines: # Empty file
                return
            # Edit first line to remove 'exposure' from headers
            if 'Exposure' in lines[0]:
                lines[0] = lines[0].replace('\tExposure' , '')
            # Edit last line to remove any carriage return that may exist
            if '\n' in lines[-1]:
                lines[-1] = lines[-1].replace('\n', '')
            with open(SAVEPATH, 'w') as f:
                f.writelines(lines)
        except FileNotFoundError:
            print(f"Error: File not found at {SAVEPATH}")
        except Exception as e:
            print(f"An error occured: {e}")
        del lines #Remove it from memory (it can be large)

    def _build_escan(self, d):
        energy_array = np.array([])
        exposure_array = np.array([])
        print(d)
        for _key, item in d.items():
            if item[2] == 0:
                energy_subset = np.array([item[0]])
            else:
                energy_subset = np.arange(item[0], item[1], item[2])
            exposure_subset = np.full_like(energy_subset, item[3])
            energy_array = np.concatenate((energy_array, energy_subset))
            exposure_array = np.concatenate((exposure_array, exposure_subset))
        return (np.round(energy_array, 2), exposure_array)

    def _iterate_dataframe(self, df, n_duplicates):
        if not isinstance(n_duplicates, int) or n_duplicates < 1:
            n_duplicates = 1

        df_duplicate = [df] * n_duplicates
        df_duplicate = pd.concat(df_duplicate, ignore_index=True)
        return df_duplicate


    def clean_slate(self):
        self.table.children = ()
        for key in (key for key in dir(self) if not key.startswith("__")):
            attr = getattr(self, key)
            if not inspect.isclass(attr) and hasattr(attr, 'variable_'):
                delattr(self, key)
