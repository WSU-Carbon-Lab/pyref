"""Resonant Soft X-ray Scattering (RSOXS) widgets for ALS beamline script generation.

This module provides widgets for generating RSOXS scan run files for
ALS beamline 11.0.1.2.
"""
# Base Packages
import datetime
import json
import os
from pathlib import Path

# Visualization Packages
import ipywidgets as widgets

# Math packages
import numpy as np
import pandas as pd
from IPython.display import display

from pyref.beamline.base_widgets import (
    ALS_ExperimentWidget,
    ALS_MeasurementWidget,
    ALS_ScriptGenWidget,
    clean_script,
    pad_digits,
)
from pyref.beamline.beamline_scan_macros import *

BASE_FILE_PATH = Path(__file__).parent

DEFAULT_ESCAN_PATH = os.path.join(BASE_FILE_PATH, "DEFAULT_ESCANS")
DEFAULT_TRAJECTORY_PATH = os.path.join(BASE_FILE_PATH, "TRAJECTORY_OPTIONS")
AVAILABLE_RSOXS_MOTORS = ['Sample Z', 'Sample Theta'] #'Sample Z', (Z removed from option)
#AVAILABLE_RSOXS_MOTORS = ['Sample X', 'Sample Y', 'Sample Z', 'Sample Theta'] #'Sample Z', (Z removed from option)


class RSOXS_Scan(ALS_MeasurementWidget):
    ALS_NAME = 'scan_'
    MEASUREMENT_TITLE = "Measurement details"
    RSOXS_MOTOR_OPTIONS = AVAILABLE_RSOXS_MOTORS
    ESCAN_OPTIONS = ['CARBON_ESCAN', 'QUICK_ESCAN']
    INITIAL_ROWS = {
            'ESCAN':'CARBON_ESCAN',
            'Sample X': 0,
            'Sample Y': 0,
            #'Sample Z': 0,
            'Sample Theta': 90
    }
    DEFAULT_TITLE = ("Scan 1",)
    FIXED_ROWS = 1
    DEFAULT_OPTIONS = {
        "fixed_title":'RSOXS Scans'
    },
    ACCORDION_SIZE = widgets.Layout(width='20rem')
    ESCAN_SIZE = widgets.Layout(width='5rem')
    HBOX_SIZE = widgets.Layout(width='50rem')

    CARBON_ESCAN = {
        "Step 1":[270.0, 280.0, 2, 0.5],
        "Step 2":[280.0, 284.0, 0.2, 1],
        "Step 3":[284.0, 286.0, 0.1, 0.5],
        "Step 4":[286.0, 292.0, 0.2, 0.5],
        "Step 5":[292.0, 300.0, 1, 0.5],
        "Step 6":[300.0, 370.0, 10, 1]
    }

    def __init__(self, save_path="",experiment_widget=None, **kwargs):
        # Import the initial saved data to create the table with
        saved_data = self.INITIAL_ROWS.copy()
        # Update from kwargs
        saved_data.update({k: v for k, v in kwargs.items() if k in saved_data})
        # Get the initial columns to populate the table with
        #self.variable_cols = [col for k, v in saved_data.items() for col in v if 'ESCAN' not in col] # Add in saved columns for the loaded data.
        # Create initial widget object
        super().__init__(experiment_widget=experiment_widget)
        # Create some generic buttons that will help duplicate/delete rows/columns
        # New title:
        self.delete_row_buttons = {}
        self.copy_row_buttons = {}
        # Initialize the table given inputs
        # Create the output that will include a title, accordion, and delete buttons
        scan_title_temp = kwargs.get('name', 'Default Name')
        self.scan_title = widgets.HTML(value="<b>Name of Sample: </b>", layout=self.LABEL_SIZE)
        self.scan_title_input = widgets.Text(value=scan_title_temp, layout=self.LABEL_SIZE)

        self.escan_details = self.generate_escan(saved_data['ESCAN']) # Create the
        self.scan_details = self.generate_accordion(saved_data)

        self.display_table = widgets.HBox(
            [
                widgets.VBox(
                    [
                        widgets.HBox([self.scan_title, self.scan_title_input]),
                        self.scan_details
                    ]
                ),
                self.escan_details
            ],
            layout=self.HBOX_SIZE
        )
    """
    Repopulate the table when changes are made:
    """

    def generate_accordion(self, d):
        scan = []
        for _i, (key, item) in enumerate(d.items()):
            if 'ESCAN' in key:
                if isinstance(item, str):
                    dropdown_item = item
                elif isinstance(item, dict):
                    dropdown_item = 'CUSTOM'
                dropdown = widgets.Dropdown(value=dropdown_item, options=self.ExperimentWidget.ScriptgenWidget.ESCAN_LIST.keys(), description=key, layout=self.ACCORDION_SIZE)
                dropdown.observe(self.update_escan, 'value') # Observe when this changes
                self.ExperimentWidget.ScriptgenWidget.dummy_widget.observe(self.escan_list_updated, 'value')
                scan.extend([dropdown])
            else:
                scan.extend([widgets.FloatText(value=item, description=key, layout=self.ACCORDION_SIZE)])
        self.scan_info = scan
        output = self.scan_info
        output.insert(0, widgets.HTML(value="<b>Scan Details:</b>"))
        return widgets.VBox(children=output)

    def generate_escan(self, input_d):
        escan_info = []
        # Check to see if the input is a dictionary, or a label
        # This handles "Saved data" or if you want ot use a default value
        if isinstance(input_d, str):
            d = self.ExperimentWidget.ScriptgenWidget.ESCAN_LIST[input_d]
        elif isinstance(input_d, dict):
            d = input_d

        title_row = widgets.HBox(
            children=[
                widgets.HTML(value="<b>Step</b>", layout=self.ESCAN_SIZE),
                widgets.HTML(value="<b>Start</b>", layout=self.ESCAN_SIZE),
                widgets.HTML(value="<b>Stop</b>", layout=self.ESCAN_SIZE),
                widgets.HTML(value="<b>Delta</b>", layout=self.ESCAN_SIZE),
                widgets.HTML(value="<b>Exposure</b>", layout=self.ESCAN_SIZE),
            ]
        )
        for _i, (key, item) in enumerate(d.items()):
            if 'Step' in key:
                temp_label = widgets.HTML(value=f"<b>{key}</b>", layout=self.ESCAN_SIZE)
                temp_start = widgets.FloatText(value=item[0], layout=self.ESCAN_SIZE)
                temp_end = widgets.FloatText(value=item[1], layout=self.ESCAN_SIZE)
                temp_step = widgets.FloatText(value=item[2], layout=self.ESCAN_SIZE)
                temp_exposure = widgets.FloatText(value=item[3], layout=self.ESCAN_SIZE)
                escan_info.extend(
                    [
                        widgets.HBox(
                            [
                                temp_label, temp_start, temp_end, temp_step, temp_exposure
                            ]
                        )
                    ]
                )
        self.escan_info = escan_info
        output = self.escan_info
        output.insert(0, title_row)
        output.insert(0, widgets.HTML(value="<b>Escan Details:</b>"))
        return widgets.VBox(output)


    def update_escan(self, change):
        """
        Updates the Escan Details widgets if the dropdown menu is changed.
        """
        new_scan = change.new # gives the new "value" of the dropdown menu
        if 'CUSTOM' in new_scan:
            return 0
        try:
            d = self.ExperimentWidget.ScriptgenWidget.ESCAN_LIST[new_scan] # Get the dictionary corresponding to the new menu
        except KeyError:
            d = CARBON_ESCAN
        self.escan_details = self.generate_escan(d)
        self.update_table()

    def escan_list_updated(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            dropdown = self.scan_info[1]
            dropdown.options = self.ExperimentWidget.ScriptgenWidget.ESCAN_LIST.keys()


    def update_table(self):
        self.display_table.children = [
                widgets.VBox(
                    [
                        widgets.HBox([self.scan_title, self.scan_title_input]),
                        self.scan_details
                    ]
                ),
                self.escan_details
            ]


    def output_dict(self):
        """
        Export relevant class values for a dictionary to save everything.
        """
        output = {}
        scan_details = self.scan_details.children
        escan_details = self.escan_details.children

        # Process scan details
        for item in scan_details:
            key = item.description
            value = item.value

            if not key:  # Skip empty description (title row)
                continue

            if 'ESCAN' in key:
                output[key] = self._process_escan_details(escan_details)
            else:
                output[key] = value

        # Add scan title
        output['name'] = self.scan_title_input.value
        return output

    def _process_escan_details(self, escan_details):
        """
        Helper function to process ESCAn details while exporting as a dictionary.
        """
        escan_data = {}

        for row in escan_details[2:]:  # Skip the first two rows (titles)
            row_values = [child.value for child in row.children]
            escan_key = row_values[0]
            escan_values = row_values[1:]

            # Check if any escan_values are zero, and skip the row if so.
            # if any(val == 0 for val in escan_values):
            #   continue

            escan_data[escan_key] = escan_values

        return escan_data


    def old_output_dict(self):
        df = {}
        for item in b.scan_details.children:
            key = item.description
            val = item.value
            if len(key) < 1: # Skip the title row
                continue
            if 'ESCAN' in key: # Cycle through the Escan values
                de = {}
                for i, row in enumerate(b.escan_details.children):
                    escan = []
                    if i < 2: # Skip the titles
                        continue
                    for i, escan_val in enumerate(row.children):
                        if i==0:
                            ekey = escan_val.value
                        else:
                            escan.append(escan_val.value)
                    for x in escan:
                        if x == 0 :
                            skip = True
                    if skip:
                        continue
                    de[ekey] = escan
                df[key] = de
            else: # Cycle through all other values
                df[key] = val
        # Add the title to deal with copy/paste elements
        df['name'] = self.scan_title_input.value
        return df


class RSOXS_Detector(ALS_ExperimentWidget):
    ALS_NAME = 'det_'
    WIDGET = RSOXS_Scan
    DEFAULT_CONSTANTS = {
        'DetX': 96.0,
        'DetY': 100.0,
        'DetTheta': 0.0127,
        'BS': 11.0,
        'SampZ': 0.0,
        'SampXOffset': 0.0,
        'SampYOffset':0.0
    }
    DEFAULT_CONSTANTS_TITLES = {
        'DetX': 'Detector X',
        'DetY': 'Detector Y',
        'DetTheta': 'CCD Theta',
        'BS': 'Beamstop',
        'SampZ': 'Sample Z',
        'SampXOffset': 'Sample X Offset',
        'SampYOffset':'Sample Y Offset'
    }
    def __init__(self, save_path="", constants=DEFAULT_CONSTANTS.copy(),constants_titles = DEFAULT_CONSTANTS_TITLES.copy(),scriptgen_widget=None, **kwargs):
        # These are defaults for the RSOXS Detector. Can be updated with **kwargs
        constants = self.DEFAULT_CONSTANTS.copy()
        constants_titles = self.DEFAULT_CONSTANTS_TITLES.copy()
        # Update from kwargs
        constants.update({k: v for k, v in kwargs.items() if k in constants})
        # Create initial widget object
        super().__init__(title='Detector Positions', constants=constants, constants_titles=constants_titles, scriptgen_widget=scriptgen_widget)

        # Buttons involved with adding rows to the RSOXS Measurement (Add items one level down)
        # This is unique to the RSOXS widget
        self.add_scan_button = widgets.Button(
            description="Add New Scan", layout=self.default_button,
            style=self.default_button_style
        )
        self.duplicate_scan_button = widgets.Button(
            description="Copy Current Scan", layout=self.default_button,
            style=self.default_button_style
        )
        self.delete_row_button = widgets.Button(
            description="Delete Current Scan", layout=self.default_button,
            style=self.default_button_style
        )
        # Add click functionality
        self.add_scan_button.on_click(lambda b: self.add_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.duplicate_scan_button.on_click(lambda b: self.copy_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.delete_row_button.on_click(lambda b: self.delete_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))

        # Update the control_buttons object
        self.control_buttons.children = [self.add_scan_button, self.duplicate_scan_button, self.delete_row_button]
        # Update the 'layout' to be a accordion that contains each scan
        self.layout = widgets.Accordion(children=[], layout=self.default_GUI)

        # Add measurements
        for key, value in kwargs.items():
            if self.WIDGET.ALS_NAME in key:
                try:
                    df = value.output_dict()
                except AttributeError:
                    df = value
                self.add_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME, **df)
        if len(self.layout.children) == 0:
            self.add_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME)

        self.GUI_experiment = widgets.VBox([self.widget_title, self.menu_box, self.control_buttons, self.layout])

    def add_scan(self, layout, widget, widget_str, b=None, **kwargs):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)])+1 # Next index
        setattr(self, widget_str + pad_digits(index), widget(experiment_widget=self, **kwargs))
        self.update_scan_tab(layout, widget_str)

    def copy_scan(self, layout, widget, widget_str, b=None):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]) + 1  # Last Index
        # Get stats of copied obj
        try:
            open_tab = self.layout.selected_index + 1
        except:
            return 0
        target_widget_name = widget_str + pad_digits(open_tab)
        if hasattr(self, target_widget_name):
            df = getattr(self, widget_str + pad_digits(open_tab)).output_dict()
        else:
            df = {}
        setattr(self, widget_str + pad_digits(index), widget(experiment_widget=self, **df))
        self.update_scan_tab(layout, widget_str)

    def delete_scan(self, layout, widget, widget_str, b=None):
        try:
            index = self.layout.selected_index + 1
        except:
            return 0
        widget_name = widget_str+pad_digits(index)
        if hasattr(self, widget_name):
            delattr(self, widget_name)
            j = index + 1
            while hasattr(self, f"{widget_str}{pad_digits(j)}"):
                setattr(self, f"{widget_str}{pad_digits(j-1)}", getattr(self, f"{widget_str}{pad_digits(j)}"))
                delattr(self, f"{widget_str}{pad_digits(j)}")
                j += 1
            self.update_scan_tab(layout, widget_str)

    def update_scan_tab(self, layout, widget_str):
        widget_list = [getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]
        display = []
        titles = []
        for widget in widget_list:
            display.extend([widget.display()])
            titles.extend([widget.scan_title_input.value])
            # Make the title change when updated:
            widget.scan_title_input.observe(self.update_title, names='value')
        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, titles[tab])

    def update_title(self, change):
        index = self.layout.selected_index
        self.layout.set_title(index, change['owner'].value) # No idea why it is setup this way.

class RSOXS_ScriptGen(ALS_ScriptGenWidget):
    EXPERIMENT_NAME = "RSoXS"
    WIDGET = RSOXS_Detector
    WIDGET_TITLE = 'Detector Position '
    JSON_TITLE = "RSOXS"
    GUI_SIZE = widgets.Layout(widget='100rem')

    MOTOR_ORDER = ['Beamline Energy', 'Sample X', 'Sample Y', 'Sample Z', 'Detector X', 'Detector Y', 'CCD Theta', 'Beamstop', 'CCD Camera Shutter Inhibit', 'Exposure']


    CARBON_ESCAN = {
        "Step 1":[270.0, 280.0, 2, 0.5],
        "Step 2":[280.0, 284.0, 0.2, 1],
        "Step 3":[284.0, 286.0, 0.1, 0.5],
        "Step 4":[286.0, 292.0, 0.2, 0.5],
        "Step 5":[292.0, 300.0, 1, 0.5],
        "Step 6":[300.0, 370.0, 10, 1]
    }
    QUICK_ESCAN = {
        "Step 1":[270.0, 280.0, 2, 0.5],
        "Step 2":[280.0, 284.0, 0.5, 1],
        "Step 3":[284.0, 286.0, 0.2, 0.5],
        "Step 4":[286.0, 292.0, 0.5, 0.5],
        "Step 5":[292.0, 300.0, 2, 0.5],
        "Step 6":[300.0, 330.0, 20, 1]
    }
    ESCAN_LIST = {
        'CUSTOM': None,
        'CARBON_ESCAN':CARBON_ESCAN,
        'QUICK_ESCAN':QUICK_ESCAN
    }

    def __init__(self, save_path="", escan_path=DEFAULT_ESCAN_PATH, **kwargs):
        super().__init__(exp_name=self.EXPERIMENT_NAME, child_widget=self.WIDGET, save_path=save_path, tab_title=self.WIDGET_TITLE,json_title=self.JSON_TITLE, **kwargs)

        # RSOXS specific details:
        self.rsoxs_button_style = {'button_color':'#5D4777', 'text_color':'white'}
        #Book-keeping attributes

        # Make additional buttons before the table
        """
        INFORMATION ON THE ESCAN DIRECTORY -- WHERE ARE JSON FILES SAVED
        """
        # Locate path for where default energy scripts are given
        self.browse_escan_directory_button = widgets.Button(
            description="Select Escan Directory", layout=self.default_button,
            style=self.rsoxs_button_style
        )
        # BUtton to update the values if the
        self.update_escan_directory_button = widgets.Button(
            description="Update Escan List", layout=self.default_button,
            style=self.rsoxs_button_style
        )
        self.browse_escan_directory_button.on_click(self.browse_escan_directory)
        self.update_escan_directory_button.on_click(self.update_escan_directory)
        self._escan_dir = ""
        self.display_escan_directory = widgets.HTML(
            value = "<b>Escan Directory: </b>" + str(self._escan_dir),
            description = ""
        )
        self.dummy_widget = widgets.Valid(value=False, description='dummy widget')

        self.escan_dir = escan_path # Update the directory with the default

        # Buttons involved with adding detector positions
        self.new_detpos_button = widgets.Button(
            description="Add New Detector Position", layout=self.default_button,
            style=self.rsoxs_button_style
        )
        self.copy_detpos_button = widgets.Button(
            description="Copy Detector Position", layout=self.default_button,
            style=self.rsoxs_button_style
        )
        self.delete_detpos_button = widgets.Button(
            description="Delete Detector Position", layout=self.default_button,
            style=self.rsoxs_button_style
        )
        # Add click functionality
        self.new_detpos_button.on_click(lambda b: self.new_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.copy_detpos_button.on_click(lambda b: self.copy_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.delete_detpos_button.on_click(lambda b: self.delete_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        # Buttons involved with adding columns to the table

        self.additional_buttons.children = (
                widgets.HBox([self.browse_escan_directory_button, self.update_escan_directory_button, self.display_escan_directory]),
                widgets.HBox([self.new_detpos_button, self.copy_detpos_button, self.delete_detpos_button])
            )

        # Select box for motor selection:
        self.select_col_title = widgets.HTML(value="<b>Select optional motors to run</b>")
        self.select_cols = widgets.SelectMultiple(
            options=AVAILABLE_RSOXS_MOTORS,
            disabled=False
        )
        self.rsoxs_menu = widgets.VBox([self.select_col_title,self.select_cols])
        self.GenerateExperiment(self.layout, self.WIDGET.ALS_NAME, self.WIDGET, **kwargs)
        self.GUI = widgets.VBox([self.title_banner, self.control_buttons, self.layout, self.json_buttons, self.rsoxs_menu, self.save_buttons], layout=self.GUI_SIZE)
        display(self.GUI)

    #@property
    #def escan_path(self):
    #    return self._escan_path
    #
    #@escan_path.setter
    #def escan_path(self, path):
    #    self._escan_path = Path(path)

    def browse_escan_directory(self, b=None):
        self.escan_dir = super().browse_directory()
        #self.update_escan_list(self.escan_dir)
        #self.display_escan_directory.value = "<b>Escan Directory:</b>" + str(self._escan_dir)
    def update_escan_directory(self, b=None):
        self.update_escan_list(self.escan_dir)

    @property
    def escan_dir(self):
        return self._escan_dir

    @escan_dir.setter
    def escan_dir(self, val):
        self._escan_dir=val
        self.display_escan_directory.value = "<b>Escan Directory:</b>" + str(self._escan_dir)
        self.update_escan_list(self._escan_dir)
        # Update the global dictionary

    def update_escan_list(self, path):
        """
        Loads all .json files in a specified path that contain 'ESCANs' for RSOXS.

        """
        output_dict = {}
        try:
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    file_path = os.path.join(path, filename)
                    with open(file_path) as f:
                        first_line = f.readline().strip() # Read the first line and remove white space
                        if first_line == "# ESCAN":
                            # Discard the first line and read the rest
                            try:
                                data = json.load(f) # Update this to ignore the title
                                file_key = os.path.splitext(filename)[0]
                                output_dict[file_key] = data
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file '{filename}': {e}")
                        else:
                            continue
        except FileNotFoundError:
            print(f"Error: Foldesr '{folder_path}' not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        self.ESCAN_LIST = self.ESCAN_LIST | output_dict
        self.dummy_widget.value = not self.dummy_widget.value # This is watched by all dropdowns to change the ESCAN_LIST when updated


    def save_script(self, ext='.txt', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'

        df = self.save_as_dict() # Get the data as a dictionary

        list_of_sample_scripts = []
        number_of_detectors = len(df) # How many detector positions?
        # Cycle through samples and start saving data
        for _i, (_index, detector) in enumerate(df.items()):
            ScanList = [s for s in detector if "scan_" in s]
            len(ScanList)
            DetectorPositions = {k: v for k, v in detector.items() if k not in ScanList}

            for _ii, scan in enumerate(ScanList):
                dm = {key: value for key, value in detector[scan].items() if key != "name"} # Drop the name which is used for saving but not scanning
                motors_to_run = ['Sample X', 'Sample Y']
                motors_to_run.extend(list(self.select_cols.value))
                #Positions = {key: value for key, value in dm.items() if key in self.select_cols.value}
                Positions = {key: value for key, value in dm.items() if key in motors_to_run}
                escan_details = self._build_escan(dm['ESCAN']) # Collect all the info for the Escan
                Escan = escan_details[0] # The energies
                Exposure = escan_details[1] # The exposures

                # Build the sample dataframe
                temp_scan = {}
                temp_scan['Beamline Energy'] = Escan
                temp_scan['Exposure'] = Exposure
                df_samp = pd.DataFrame(temp_scan)

                # Add the sample positions
                for motor, pos in Positions.items():
                    df_samp[motor] = np.full_like(Escan, pos)
                # Add detector Positions
                if number_of_detectors > 1:
                    for motor, pos in DetectorPositions.items():
                        if "DetX" in motor:
                            key = 'Detector X'
                        elif "DetY" in motor:
                            key = 'Detector Y'
                        elif "DetTheta" in motor:
                            key = 'CCD Theta'
                        elif "BS" in motor:
                            key = 'Beamstop'
                        elif "SampXOffset" in motor:
                            df_samp['Sample X'] += pos
                            continue
                        elif "SampYOffset" in motor:
                            df_samp['Sample Y'] += pos
                            continue
                        df_samp[key] = np.full_like(Escan, pos)
                # Add CCD Inhibit, first and last, all exposure times
                df_samp['CCD Camera Shutter Inhibit'] = np.full_like(Escan, 0) # Add the column
                for expo in np.unique(Exposure):
                    df_samp = pd.concat([df_samp.loc[0].to_frame().T, df_samp], ignore_index=True) # Add a frame at the beginning of the sample
                    df_samp = pd.concat([df_samp, df_samp.loc[df_samp.index[-1]].to_frame().T], axis=0, ignore_index=True) # Add a frame at the end of the sample

                    df_samp.loc[0, 'CCD Camera Shutter Inhibit'] = 1 # Close the shutter
                    df_samp.loc[df_samp.index[-1], 'CCD Camera Shutter Inhibit'] = 1

                    df_samp.loc[0, 'Exposure'] = expo # Set exposure
                    df_samp.loc[df_samp.index[-1], 'Exposure'] = expo
                list_of_sample_scripts.append(df_samp)

        RunFile = pd.concat(list_of_sample_scripts, ignore_index=True) # Only export with the Adjustable Motors in the correct order.

        # Cleanup the output because the ALS requires specific things
        cols = list(RunFile.columns)
        Ordered_Cols = [item for item in self.MOTOR_ORDER if item in cols]
        RunFile = RunFile[Ordered_Cols]

        RunFile.to_csv(SAVEPATH, index=False, sep='\t')

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
        for _key, item in d.items():
            if item[2] == 0:
                energy_subset = np.array([item[0]])
            else:
                energy_subset = np.arange(item[0], item[1], item[2])
            exposure_subset = np.full_like(energy_subset, item[3])
            energy_array = np.concatenate((energy_array, energy_subset))
            exposure_array = np.concatenate((exposure_array, exposure_subset))
        return (np.round(energy_array, 2), exposure_array)



def update_dict(old_dict, new_dict):
    for key in old_dict:
        if key in new_dict:
            old_dict[key] = new_dict[key]
    return old_dict

def clean_for_widgets(my_dict):
    for key, value in my_dict.items():
        if isinstance(value, list):
            my_dict[key] = ', '.join(map(str, value))
    return my_dict



"""
****************THESE ARE MACRO SCANS. VERY SIMILAR BUT WILL SAVE AS BEAMLINE SCAN FILES
"""

class RSOXS_Detector_Macro(RSOXS_Detector):

    DEFAULT_CONSTANTS = {
        'SampXOffset': 0.0,
        'SampYOffset':0.0,
        'Trajectory': "Photodiode_Far"
    }
    DEFAULT_CONSTANTS_TITLES = {
        'SampXOffset':'Sample X Offset',
        'SampYOffset':'Sample Y Offset',
        'Trajectory' : 'Trajectory'
    }

    def __init__(self, save_path="", scriptgen_widget=None, **kwargs):

        constants = self.DEFAULT_CONSTANTS.copy()
        constants_titles = self.DEFAULT_CONSTANTS_TITLES.copy()
        # Update from kwargs
        constants.update({k: v for k, v in kwargs.items() if k in constants})
        super().__init__(save_path=save_path, scriptgen_widget=scriptgen_widget, constants=constants, constants_titles=constants_titles, **kwargs)
        self.GUI_experiment.children = () # Empty the GUI to repopulate it for the MACRO

        # New content for specific trajectories --

        # Dummy widget used to update dropdown if things change
        self.dummy_widget = widgets.Valid(value=False, description='dummy widget') # Does not exist
        self.dummy_widget.observe(self.trajectory_list_updated, 'value')

        self._trajectory_dir = "" # Initialize the value
        # Info to describe trajectory
        self.trajectory_info = widgets.HTML(value = "<i>Trajectory Directory: </i>", style={'description_width': '160px'}) # Title
        self.display_trajectory_directory = widgets.HTML(str(self._trajectory_dir)) # Display widget
        self.trajectory_dir = DEFAULT_TRAJECTORY_PATH # Set the value using @property

        # Button to change the trajectory menu. Not usually in use -- May remove in future update
        self.browse_trajectory_directory_button = widgets.Button(
            description="Select Directory",
            layout = self.default_button_style
        )
        self.browse_trajectory_directory_button.on_click(self.browse_trajectory_directory) # Button functionality
        # Storage Box that contains all Trajectory related information --
        self.trajectory_directory_items = widgets.VBox([widgets.HBox([self.trajectory_info,self.display_trajectory_directory]), self.browse_trajectory_directory_button])

        # Dropdown used to select trajectory associated with the detector position
        # This will override what is originally created as a FloatText option
        self.Trajectory = widgets.Dropdown(
            value = constants['Trajectory'],
            options = self.ScriptgenWidget.TRAJECTORY_LIST.keys(),
            layout = widgets.Layout(width='30rem'),
            description = 'Select Detector Trajectory',
            style = {'description_width':'10rem'}
        )

        # Dummy widget to update TRAJECTORY_LIST

        self.GUI_experiment = widgets.VBox([self.widget_title, self.trajectory_directory_items,self.Trajectory, self.menu_box, self.control_buttons, self.layout])


    def browse_trajectory_directory(self, b):
        self.trajectory_dir = self.ScriptgenWidget.super().browse_directory()

    @property
    def trajectory_dir(self):
        return self._trajectory_dir
    @trajectory_dir.setter
    def trajectory_dir(self, val):
        self._trajectory_dir = val
        self.display_trajectory_directory.value = str(self._trajectory_dir)
        self.update_trajectory_list(val)

    def update_trajectory_list(self, path):
        """
        Loads all .json files in a specified path that contain 'ESCANs' for RSOXS.

        """
        output_dict = {}
        try:
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    file_path = os.path.join(path, filename)
                    with open(file_path) as f:
                        first_line = f.readline().strip() # Read the first line and remove white space
                        if first_line == "# Trajectory":
                            # Discard the first line and read the rest
                            try:
                                data = json.load(f) # Update this to ignore the title
                                file_key = os.path.splitext(filename)[0]
                                output_dict[file_key] = data
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in file '{filename}': {e}")
                        else:
                            continue
        except FileNotFoundError:
            print(f"Error: Foldesr '{path}' not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        self.ScriptgenWidget.TRAJECTORY_LIST = output_dict
        self.dummy_widget.value = not self.dummy_widget.value # This is watched by all dropdowns to change the ESCAN_LIST when updated

    def trajectory_list_updated(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            dropdown = self.Trajectory
            dropdown.options = self.ScriptgenWidget.TRAJECTORY_LIST.keys()

class RSOXS_MacroGen(RSOXS_ScriptGen):
    WIDGET = RSOXS_Detector_Macro
    TRAJECTORY_LIST = {}

    def __init__(self, save_path="", **kwargs):
        super().__init__(save_path=save_path, **kwargs)
        self.GUI.children = () # Clean out the default GUI options to make room for the new version
        # New Banner for the MACRO generator
        self.title_banner = widgets.HTML(value=f"<h2>{self.EXPERIMENT_NAME} Macro Generator</h2>")

        # All buttons are the same as ScriptGen except the MacroGen ---

        # NEW BUTTONS THAT HELP SAVE THE BEAMLINE MACRO
        # New layouts and styles for the macro-specific options
        self.macro_button_layout = widgets.Layout(widget='50rem')
        self.macro_style = {'description_width': '160px'}
        self.macro_box_layout = widgets.Layout(widget='60rem')
        self.macro_title_style = {'font_size': '14px'}
        self.macro_info_style = {'font_size': '12px'}

        # TItle for Macro options --
        self.macro_title = widgets.HTML(value="<b>Options for Beamline Scan Macro</b>", style=self.macro_title_style)
        """
        Build the Spiral Scan menu for Spiral Scans -- Use for preliminary
        """
        self.spiral_tab = self.build_spiral_tab()
        self.nexafs_tab = self.build_nexafs_tab()
        self.i0_tab = self.build_i0_tab()

        self.AdditionalMacroOptions = widgets.Tab(
            children = [self.spiral_tab, self.nexafs_tab, self.i0_tab],
            layout = self.default_GUI,
            titles = ['Spiral Scan', 'NEXAFS', 'I0 Options']
        )

        self.ADDITIONAL_MACRO_BUTTONS = widgets.VBox(
            [
                self.macro_title,
                self.AdditionalMacroOptions
            ],
            layout=self.macro_box_layout
        )

        self.GenerateExperiment(self.layout, self.WIDGET.ALS_NAME, self.WIDGET, **kwargs)
        self.GUI = widgets.VBox([self.title_banner, self.control_buttons, self.layout, self.json_buttons, self.rsoxs_menu, self.ADDITIONAL_MACRO_BUTTONS, self.save_button_beamline], layout=self.GUI_SIZE)
        display(self.GUI)


    def save_script(self, ext='.txt', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEDIR + '/' + SAVENAME + '.txt'

        df = self.save_as_dict() # Get the data as a dictionary
        macro_details = self._build_individual_scan_files(df)

        # Collect i0 details for broader macro
        i0_condition = self.I0_select.value
        i0_posx = self.I0_xpos.value
        i0_posy = self.I0_ypos.value
        i0_escan = self.I0_escan.value
        if i0_condition != 'Never': # Waste time building a scan?
            i0_script = SAVEDIR + '/' + "I0_script.txt"
            self._save_escan_script(i0_script, self.ESCAN_LIST[i0_escan], exposure=0.5)

        # START BUILDING MACRO
        scans_elapsed = 0 # If I0s are to be taken after a certain number of scans
        macro = begin_macro(SAVENAME) # list of commands that will become the beamline scan
        if "Before" in i0_condition:
            macro.extend(run_I0("I0_BeforeMacro_", i0_script, i0_posx, i0_posy))
        if not self.only_run_nexafs.value: # If you only want NEXAFS it skips the RSOXS portion
            macro += [set_instrument("CCD", exp=0.1, extension='fits')] # Turn on CCD for data collection
            for i, det in enumerate(macro_details):
                # Check to see if we want an I0 before we go to the next detector position
                if "detector" in i0_condition:
                    macro.extend(run_I0("I0_BeforeDet_"+str(i), i0_script, i0_posx, i0_posy))
                    macro += [set_instrument("CCD", exp=0.1, extension='fits')] # Turn on CCD for data collection after I0
                # First thing is to go to the detector position
                macro.extend(run_dict_trajectory(det['Trajectory']))
                # Start cycling through samples
                for sample in det['sample_scripts']:
                    macro += [analog_from_file(sample[0], sample[1])]
                    scans_elapsed += 1 # Keep track of scans elapsed
                    # Interupt and run an I0 if requested
                    if "X scans" in i0_condition:
                        if scans_elapsed % self.i0_X.value == 0:
                            macro.extend(run_I0("I0_AfterScan_"+str(scans_elapsed), i0_script, i0_posx, i0_posy))
                            macro += [set_instrument("CCD", exp=0.1, extension='fits')] # Reinitialize after I0
                            macro.extend(run_dict_trajectory(det['Trajectory']))
            if "After" in i0_condition:
                macro.extend(run_I0("I0_AfterMacro", i0_script, i0_posx, i0_posy))
        macro += [clear_instruments] # Go to the Photodiode only
        macro.extend(run_dict_trajectory(Photodiode_Far)) # Go to the photodiode position
        if self.post_run_nexafs.value: # Run NEXAFS after cycle?
            macro += [add_comment("Running NEXAFS")]
            NEXAFS_script = SAVEDIR + '/' + "NEXAFS_script.txt"
            dnexafs = df[f'det_{pad_digits(1)}']
            ScanList = [s for s in dnexafs if "scan_" in s] # Get all the scans for the first detector position
            for sample in ScanList:
                macro += [move_motor('Sample X', dnexafs[sample]['Sample X'], 0.1)]
                macro += [move_motor('Sample Y', dnexafs[sample]['Sample Y'], 0.1)]

                if self.nexafs_escan != "CUSTOM":
                    self._save_escan_script(NEXAFS_script, self.ESCAN_LIST[self.nexafs_escan.value], exposure=0.5)
                else:
                    self._save_escan_script(NEXAFS_script, dnexafs[sample]['ESCAN'], exposure=0.5)

                macro += [analog_from_file(dnexafs[sample]['name'] + "_NEXAFS", NEXAFS_script)]
        macro.extend(run_I0("I0_ScanComplete", i0_script, i0_posx, i0_posy))
        macro += [add_prompt("Finished Macro")]

        build_macro(SAVENAME, SAVEDIR, macro)


    def save_spiral(self, ext='.txt.', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEDIR + '/' + SAVENAME + '_spiral.txt'

        df = self.save_as_dict() # Get the data as a dictionary
        macro_details = self._build_spiral_scan(df)

        macro = begin_macro(SAVENAME) # list of commands that will become the spiral scan
        macro += [set_instrument("CCD", exp=0.1, extension='fits')] # Turn on CCD for data collection
        macro.extend(run_dict_trajectory(macro_details['Trajectory'])) # Move to the position where data is collected
        for sample in macro_details['spiral_scripts']:
            macro += [analog_from_file(sample[0], sample[1])]
        macro += [add_prompt("Finished Macro")]
        build_macro(SAVENAME, SAVEDIR, macro)


    def _build_individual_scan_files(self, df):
        # Initialize the macro construction
        list_of_detectors = [] # Contains lists associated with each detector position
        number_of_detectors = len(df) # How many detector positions?

        # Where the data will be saved
        MACRO_DIR = str(self.save_dir)
        """
        Cycle through the detector positions and start saving scripts
        Each script will be measured on one sample, and one detector position.
        Current naming convention: {name}_det{X}.txt
        """
        # Cycle through detector positions and start saving scripts
        for i, (_index, detector) in enumerate(df.items()):
            # Initialize storage for each detector
            detector_details = {} # Dictionary that contains all details pertaining to the current detector position
            all_sample_scripts = [] # List that will contain all sample dataframes for the current detector
            # Start collecting data --
            ScanList = [s for s in detector if "scan_" in s] # Get all the scans in the current detector position
            # number_of_samples = len(ScanList) # How many samples in total?
            DetectorPositions = {k: v for k, v in detector.items() if k not in ScanList} # All items relevant to setting up the detector geometry

            # Start compiling details for the detector
            detector_details['Trajectory'] = self.TRAJECTORY_LIST[DetectorPositions['Trajectory']]

            # Run through each ScanList and build the individual scan files for each sample --
            for ii, scan in enumerate(ScanList):
                dm = {key: value for key, value in detector[scan].items() if key != "name"} # Drop the name which is used for saving but not scanning
                sample_name = detector[scan]['name']
                motors_to_run = ['Sample X', 'Sample Y'] # Define the sample position
                if 'Sample Theta' in self.select_cols.value:
                    motors_to_run.extend(['Sample Theta']) # Add Theta if you want to include it

                # Get motor positions to cycle through
                Positions = {key: value for key, value in dm.items() if key in motors_to_run}
                # Build the Escan based on RSOXS conventions
                escan_details = self._build_escan(dm['ESCAN']) # Collect all the info for the Escan
                Escan = escan_details[0] # Get energies
                Exposure = escan_details[1] # Get exposures

                # Build the sample dataframe
                temp_scan = {}
                temp_scan['Beamline Energy'] = Escan
                temp_scan['Exposure'] = Exposure
                df_samp = pd.DataFrame(temp_scan)

                # Add the sample positions
                for motor, pos in Positions.items():
                    df_samp[motor] = np.full_like(Escan, pos)
                # Add detector Positions
                if number_of_detectors > 1:
                    for motor, pos in DetectorPositions.items():
                        if "SampXOffset" in motor and 'Sample X' in Positions:
                            df_samp['Sample X'] += pos
                            continue
                        elif "SampYOffset" in motor and 'Sample Y' in Positions:
                            df_samp['Sample Y'] += pos
                            continue
                        #df_samp[key] = np.full_like(Escan, pos)
                # Add CCD Inhibit, first and last, all exposure times
                df_samp['CCD Camera Shutter Inhibit'] = np.full_like(Escan, 0) # Add the column
                for expo in np.unique(Exposure):
                    df_samp = pd.concat([df_samp.loc[0].to_frame().T, df_samp], ignore_index=True) # Add a frame at the beginning of the sample
                    df_samp = pd.concat([df_samp, df_samp.loc[df_samp.index[-1]].to_frame().T], axis=0, ignore_index=True) # Add a frame at the end of the sample

                    df_samp.loc[0, 'CCD Camera Shutter Inhibit'] = 1 # Close the shutter
                    df_samp.loc[df_samp.index[-1], 'CCD Camera Shutter Inhibit'] = 1

                    df_samp.loc[0, 'Exposure'] = expo # Set exposure
                    df_samp.loc[df_samp.index[-1], 'Exposure'] = expo

                # Cleanup the script because the ALS requires specific things
                cols = list(df_samp.columns)
                Ordered_Cols = [item for item in self.MOTOR_ORDER if item in cols]
                df_samp = df_samp[Ordered_Cols]

                # save the df_samp as a macro file
                SAVE_SCAN = MACRO_DIR + '/' + sample_name +'_det'+str(i) + '_escan'+str(ii) + '.txt'
                df_samp.to_csv(SAVE_SCAN, index=False, sep='\t')
                clean_script(SAVE_SCAN) # Remove exposure and clear last line if it exists

                all_sample_scripts.append((sample_name, SAVE_SCAN)) # Save path for macro generation

            detector_details['sample_scripts'] = all_sample_scripts # Collect all scripts under the detector
            list_of_detectors.append(detector_details)
        return list_of_detectors


    def _build_spiral_scan(self, df):
        # Where will I save these files?
        MACRO_DIR = str(self.save_dir)

        list_of_scripts = [] # A compiled list of script paths for macro generation
        # pick the detector that you want to run your spiral scan with
        detector_choice = self.spiral_scan_det.value
        # Make a list of energies that you want to run the spiral scan with
        energy_float= [float(item.strip()) for item in self.spiral_scan_energy.value.split(',')]
        # Get step size and exposure
        spiral_step = self.spiral_scan_step.value
        grid_size = int(self.spiral_scan_grid_size.value)
        exposure_choice = self.spiral_scan_exposure.value

        # Cycle through samples and start saving data
        try:
            detector = df[f"det_{pad_digits(detector_choice)}"]
        except:
            detector = df[f"det_{pad_digits(1)}"] # Default to the first one (will always exist)

        # Start getting scans
        ScanList = [s for s in detector if "scan_" in s]
        len(ScanList)
        DetectorPositions = {k: v for k, v in detector.items() if k not in ScanList}

        detector_details = {}
        detector_details['Trajectory'] = self.TRAJECTORY_LIST[DetectorPositions['Trajectory']]
        #if 'Sample Z' in self.select_cols.value:
        #    detector_details['Sample Z'] = DetectorPositions['SampZ']

        for i, scan in enumerate(ScanList):
            sample_grid_df = [] # Where we will store the sample position scans
            # Drop the name which is used for saving but not scanning
            {key: value for key, value in detector[scan].items() if key != "name"}
            sample_name = detector[scan]['name'] # Name to save the script

            Escan = np.array(energy_float) # Collect all the info for the Escan
            Exposure = np.full_like(Escan, exposure_choice)

            # Build the sample dataframe
            temp_scan = {}
            temp_scan['Beamline Energy'] = Escan
            temp_scan['Exposure'] = Exposure
            df_temp = pd.DataFrame(temp_scan)

            # Sample Positions
            center_x = detector[scan]['Sample X']
            center_y = detector[scan]['Sample Y']
            # Create grid
            grid = generate_2d_grid(center_x, center_y, spiral_step, grid_size=grid_size)

            for _ii, point in enumerate(grid):
                df_grid = df_temp.copy()
                df_grid['Sample X'] = np.full_like(Escan, point[0])
                df_grid['Sample Y'] = np.full_like(Escan, point[1])
                df_grid['CCD Camera Shutter Inhibit'] = np.full_like(Escan, 0) # Add the column

                # Cleanup the script because the ALS requires specific things
                cols = list(df_grid.columns)
                Ordered_Cols = [item for item in self.MOTOR_ORDER if item in cols]
                df_grid = df_grid[Ordered_Cols]
                sample_grid_df.append(df_grid)

            df_samp = pd.concat(sample_grid_df, ignore_index=True) # Combine all samples into a single scan to run
            df_out = df_samp.sort_values(by=['Beamline Energy']) # Only move the undulator once or twice

            # save the df_samp as a macro file
            SAVE_SCAN = MACRO_DIR + '/' + sample_name +'_det'+str(detector_choice) + '_spiral'+str(i) + '.txt'
            df_out.to_csv(SAVE_SCAN, index=False, sep='\t')
            clean_script(SAVE_SCAN) # Remove exposure and clear last line if it exists

            list_of_scripts.append((sample_name+"_spiral"+str(i),SAVE_SCAN)) # Save the path
        detector_details['spiral_scripts'] = list_of_scripts
        return detector_details

    def _save_escan_script(self, path, d, exposure=None):
        # Compile Details into exposure and energies
        escan_details = self._build_escan(d)
        # Build a dictionary
        as_dict = {}
        as_dict['Beamline Energy'] = escan_details[0]
        if exposure is None:
            as_dict['Exposure'] = escan_details[1]
        else:
            as_dict['Exposure'] = np.full_like(escan_details[1], exposure)
        # Make a dataframe for easy saving
        df = pd.DataFrame(as_dict)
        df.to_csv(path, index=False, sep='\t')
        clean_script(path) # Remove exposure and clear last line if it exists



    def build_spiral_tab(self):
        self.spiral_scan_title = widgets.HTML(value="<b>Spiral Scan Options</b>", style=self.macro_title_style)
        self.spiral_scan_info = widgets.HTML(
            value = "<i>Options to generate a spiral scan macro.<br>Will measure short scans in X-by-X grid around sample position<br>Use to determine optimum scan location</i>",
            style = self.macro_info_style
        )

        self.spiral_scan_button = widgets.Button(
            description='Save Spiral Scan Macro',
            style = self.macro_style,
            layout = self.macro_button_layout
        )
        self.spiral_scan_button.on_click(self.save_spiral)
        self.spiral_scan_energy = widgets.Text(
            value = '270.0, 284.0',
            description = "Spiral Scan Energies (eV):",
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.spiral_scan_step = widgets.FloatText(
            value = 0.300,
            description = 'Spiral Step Size (mm)',
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.spiral_scan_grid_size = widgets.BoundedIntText(
            value = 5,
            min=1,
            max=7,
            step=2,
            description = 'Grid Size (X-by-X)',
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.spiral_scan_det = widgets.IntText(
            value = 1,
            description = "Detector Position: ",
            style = self.macro_style,
            layout = self.macro_button_layout
        )
        self.spiral_scan_exposure = widgets.FloatText(
            value = 1.0,
            description = 'Spiral Scan Exposure (s): ',
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.spiral_options = widgets.VBox(
            [
                self.spiral_scan_title,
                self.spiral_scan_info,
                self.spiral_scan_energy,
                self.spiral_scan_exposure,
                self.spiral_scan_step,
                self.spiral_scan_grid_size,
                self.spiral_scan_det,
                self.spiral_scan_button,
            ],
            layout=self.macro_box_layout
        )
        return self.spiral_options

    def build_nexafs_tab(self):
        self.nexafs_title = widgets.HTML(value="<b>NEXAFS Options</b>", style=self.macro_title_style)
        #self.nexafs_title = widgets.HTML(value="<b>Include Transmission NEXAFS</b>", style=self.macro_title_style)
        self.nexafs_info = widgets.HTML(value = "<i>Options to run NEXAFS in addition to RSOXS<br>Highly recommended for all transmission RSOXS measurements.</i>", style=self.macro_info_style)

        self.post_run_nexafs = widgets.Checkbox(
            value=False,
            description='Include NEXAFS scan?',
            disabled=False,
            indent=False,
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.only_run_nexafs = widgets.Checkbox(
            value=False,
            description='Only run NEXAFS? (Will not initialize the CCD)',
            disabled=False,
            indent=False,
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.nexafs_info2 = widgets.HTML(value = "<i>What transmission scan do you want to run?<br>Select CUSTOM to repeat energy scan used in each RSOXS measurement</i>", style=self.macro_info_style)

        self.nexafs_escan = widgets.Dropdown(
            value ='QUICK_ESCAN',
            options = self.ESCAN_LIST.keys(),
            description="Scan File: ",
            style = self.macro_style,
            layout=self.macro_button_layout

        )
        self.nexafs_options = widgets.VBox(
            [
                self.nexafs_title,
                self.nexafs_info,
                self.post_run_nexafs,
                self.only_run_nexafs,
                self.nexafs_info2,
                #self.nexafs_info3,
                self.nexafs_escan
            ],
            layout=self.macro_box_layout
        )
        return self.nexafs_options

    def build_i0_tab(self):
        self.I0_title = widgets.HTML(value="<b>Direct Beam (I0) Options</b>", style=self.macro_title_style)
        self.I0_info = widgets.HTML(value="<i>Select when you want to run an I0 during macro</i>")
        self.I0_list = [
            "Never",
            "Before macro",
            "Before and after macro",
            "Start of each detector position",
            "After every X scans"
        ]
        self.I0_select = widgets.Dropdown(
            options = self.I0_list,
            value = "Before and after macro",
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.I0_xpos = widgets.FloatText(
            value = 0.0,
            description = "I0 X position",
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.I0_ypos = widgets.FloatText(
            value = 0.0,
            description = "I0 Y position",
            style = self.macro_style,
            layout=self.macro_button_layout
        )
        self.I0_escan = widgets.Dropdown(
            value ='CARBON_ESCAN',
            options = self.ESCAN_LIST.keys(),
            description="Scan File: ",
            style = self.macro_style,
            layout=self.macro_button_layout

        )
        self.I0_options = widgets.VBox(
            [
                self.I0_title,
                self.I0_info,
                self.I0_select,
                self.I0_xpos,
                self.I0_ypos,
                self.I0_escan
            ],
            layout=self.macro_box_layout
        )
        return self.I0_options







def update_dict(old_dict, new_dict):
    for key in old_dict:
        if key in new_dict:
            old_dict[key] = new_dict[key]
    return old_dict

def clean_for_widgets(my_dict):
    for key, value in my_dict.items():
        if isinstance(value, list):
            my_dict[key] = ', '.join(map(str, value))
    return my_dict


def generate_2d_grid(center_x, center_y, step_size, grid_size=5):
    """
    Generates a 2D grid of x and y coordinates around a center point.

    Args:
        center_x: The x-coordinate of the center point.
        center_y: The y-coordinate of the center point.
        step_size: The distance between grid points.
        grid_size: The size of the grid (e.g., 5 for a 5x5 grid).

    Returns
    -------
        A list of tuples, where each tuple represents (x, y) coordinates.
    """
    grid = []
    half_grid = grid_size // 2

    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            x = center_x + i * step_size
            y = center_y + j * step_size
            grid.append((x, y))

    return grid



def get_current_datetime_string():
    """Returns a string with the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


