"""Polarized X-ray Reflectivity (PXR) widgets for ALS beamline script generation.

This module provides widgets for generating PXR scan run files for
ALS beamline 11.0.1.2.
"""
# Base Packages
import sys
import os
import json
import re
import copy
import inspect
from pathlib import Path

# Math packages
import numpy as np
import pandas as pd

# Visualization Packages
import ipywidgets as widgets
from IPython.display import display, clear_output
import tkinter as tk
from tkinter import filedialog

from pyref.beamline.base_widgets import (
    ALS_ScriptGenWidget,
    ALS_ExperimentWidget,
    ALS_MeasurementWidget,
    clean_script,
    pad_digits,
)

from pyref.beamline.beamline_scan_macros import *


# Variables that might be adjusted later as needed

X_VARIABLES = [
    "Qval",
    "Theta"
]

POINT_DENSITY_METHOD = [
    'Use Thickness',
    'Use Delta'
]

AVAILABLE_PXR_MOTORS = [
    "Upstream JJ Vert Aperture",
    "Upstream JJ Horz Aperture",
    "Horizontal Exit Slit"
]
AVAILABLE_PXR_MOTOR_DEFAULTS = {
    "Upstream JJ Vert Aperture": 0.08,
    "Upstream JJ Horz Aperture": 0.08,
    "Horizontal Exit Slit": 3000
}

class PXR_Scan(ALS_MeasurementWidget):
    ALS_NAME = 'scan_'
    LAYOUR_FLOATTEXT = widgets.Layout(width='300px')
    STYLE_FLOATTEXT = {'description_width': '150px'}
    LABEL_SIZE = widgets.Layout(width='150px')
    CELL_SIZE = widgets.Layout(width='100px')
    DEFAULT_TITLE = "Meausrement Details"
    DEFAULT_PARAMETERS = {
        'theta_end': 60.0,
        'energy_start': 250.0,
        'energy_stop': 280.0,
        'energy_delta': 0,
        'Sample Theta': [1,5,10,15,20,30,40],
        'Higher Order Suppressor': [10, 9, 8.5, 7.5, 7.5, 7.5, 7.5],
        'Exposure': [0.0015, 0.0015, 0.0015, 0.0015, 0.1, 0.5, 1]
    }
    def __init__(self, experiment_widget=None, **kwargs):
        parameters = self.DEFAULT_PARAMETERS.copy()
        #constants_titles = self.DEFAULT_CONSTANTS_TITLES.copy()
        # Update those based on what the kwargs are -- 
        parameters.update({k: v for k, v in kwargs.items() if k in parameters})
        super().__init__(constant_motor_title='Measurement Details', experiment_widget=experiment_widget) # Build the initial table

        VAR_KEY = 'Sample Theta'
        total_cols = len(parameters[VAR_KEY])

        # Initialize table given inputs
        self.table_titles = [
            [widgets.HTML(value="<b>Motor to move:</b>", layout=self.LABEL_SIZE)] +
            [widgets.HTML(value="<b>Step "+str(i+1)+"</b>", layout=self.CELL_SIZE) for i in range(total_cols)]
        ]
        # Make the initial rows
        for i, (key, item) in enumerate(parameters.items()):
            if isinstance(item, list):
                if key in ['Sample Theta', 'Higher Order Suppressor', 'Exposure']: # The big 3
                    setattr(
                        self,
                        'fixed_'+str(i),
                        [
                            [widgets.Label(value=key, layout=self.LABEL_SIZE)] + 
                            [widgets.FloatText(value=ii, layout=self.CELL_SIZE) for ii in item]
                        ]
                    )
                else:
                    setattr(
                        self,
                        'variable_'+str(i),
                        [
                            [widgets.Label(value=key, layout=self.LABEL_SIZE)] +
                            [widgets.FloatText(value=ii, layout=self.CELL_SIZE) for ii in item]
                        ]
                    )
        # All samples have the following objects
        # Energy Scan information
        self.energy_start = widgets.FloatText(
            value = parameters['energy_start'],
            description = "Start : ",
            layout = self.LAYOUR_FLOATTEXT,
            style = self.STYLE_FLOATTEXT
        )
        self.energy_stop = widgets.FloatText(
            value = parameters['energy_stop'],
            description = "Stop: ",
            layout = self.LAYOUR_FLOATTEXT,
            style = self.STYLE_FLOATTEXT
        )
        self.energy_delta = widgets.FloatText(
            value = parameters['energy_delta'],
            description = "Step: ",
            layout = self.LAYOUR_FLOATTEXT,
            style = self.STYLE_FLOATTEXT
        )
        self.energy_title = widgets.HTML(value="Photon Energy Scan [eV]")
        self.energy_desc = widgets.HTML(value="<i>Set delta=0 to run single energy</i>")
        
        self.energy_info = widgets.VBox([self.energy_title, self.energy_desc])
        self.theta_end = widgets.FloatText(
            value = parameters['theta_end'],
            description = "Final Sample Theta: ",
            layout = widgets.Layout(width='px'),
            style = self.STYLE_FLOATTEXT
        )
        # Compile all values that are fixed and apply to the full title
        self.constant_motor_attrs = ['energy_start','energy_stop','energy_delta', 'theta_end']
        self.table = widgets.VBox(self.update_table())
        
        # Create a button to add or remove columns and rows
        self.add_step_button = widgets.Button(description="Add step")
        self.add_step_button.on_click(self.add_step)

        self.remove_step_button = widgets.Button(description="Remove step")
        self.remove_step_button.on_click(self.remove_step)

        # Create a button to add rows
        self.add_motor_button = widgets.Button(description="Add motor")
        self.add_motor_button.on_click(self.add_motor)

        self.remove_motor_button = widgets.Button(description="Remove motor")
        self.remove_motor_button.on_click(self.remove_motor)

        # Put them into the control buttons item
        self.control_buttons.children = [
            widgets.HBox(
                [
                    self.add_step_button, self.add_motor_button
                ]
            ),
            widgets.HBox(
                [
                    self.remove_step_button, self.remove_motor_button
                ]
            )
        ]
        self.display_table = self.build_display_table()
        #display(self.display_table)
        
    def build_display_table(self):
        build_table = []
        build_table.extend([self.constant_motor_title]) # The title of fixed values
        build_table.extend([self.energy_info])
        build_table.extend([getattr(self, widget) for widget in self.constant_motor_attrs]) # Load in the values
        build_table.extend([self.table])  # table object if needed
        build_table.extend([self.control_buttons]) # optional buttons
        return widgets.VBox(build_table)

    def add_step(self, b):
        new_step = len(self.table_titles[0]) # Determine the next step digit
        motor_directory = self.get_table(include_title=True)
        for motor in motor_directory:
            if "title" in motor:
                getattr(self, motor)[0].append(
                    widgets.HTML(
                        value="<b>Step "+str(new_step)+"</b>",
                        layout=self.CELL_SIZE
                    )
                )
            else:
                getattr(self,motor)[0].append(
                    widgets.FloatText(
                        value=0.0,
                        layout=self.CELL_SIZE
                    )
                )

        # Update the table
        self.table.children = self.update_table()

    def remove_step(self, b):
        final_step = len(self.table_titles[0]) # Determine the final step
        # No steps to remove
        if final_step == 1:
            print("No steps to delete")
            return 0
        motor_directory = self.get_table(include_title=True)
        for motor in motor_directory:
            getattr(self,motor)[0].pop()  
       
        # Update the table
        self.table.children = self.update_table()

    def add_motor(self, b=None):
        total_steps = len(self.table_titles[0])-1
        # Add a new row with FloatText widgets
        num_extra_motors = len([attr for attr in dir(self) if attr.startswith("variable_")])
        if num_extra_motors >= len(AVAILABLE_PXR_MOTORS):
            return 0
        # Create the new attribute
        setattr(
            self,
            "variable_"+str(num_extra_motors+1),
            [
                #[widgets.Text(value="", layout=self.label_size)] +
                [widgets.Dropdown(options=AVAILABLE_PXR_MOTORS, value=AVAILABLE_PXR_MOTORS[0], layout=self.LABEL_SIZE)] +
                [widgets.FloatText(value=AVAILABLE_PXR_MOTOR_DEFAULTS[AVAILABLE_PXR_MOTORS[0]], layout=self.CELL_SIZE) for i in np.arange(total_steps)]
                #[widgets.Dropdown(options=AVAILABLE_MOTORS, value=::)]
            ]            
        )
        temp_motor = getattr(self, "variable_"+str(num_extra_motors+1))
        self.table.children = self.update_table()

    def remove_motor(self, b=None):
        num_extra_motors = len([attr for attr in dir(self) if attr.startswith("variable_")])
        for motor in self.get_table():
            if "variable_" in motor:
                if str(num_extra_motors) in motor:
                    delattr(self,motor)
        
        # Update the table layout
        self.table.children = self.update_table()
        

class PXR_Experiment(ALS_ExperimentWidget):
    ALS_NAME = 'exp_'
    WIDGET = PXR_Scan 
    DEFAULT_CONSTANTS = {
        'XPosition': 0.0,
        'YPosition': 0.0,
        'ZPosition': 0.0,
        'ZFlipPosition': 0.0,
        'DirectBeam': -5.0,
        'XOffset': 0.15,
        'ReverseHolder': False,
        'ThetaOffset': 0.0,
        'IndependentVariable': 'Qval',
        'DensityCalc': 'Use Thickness',
        'SampleThickness': 250,
        'AngleCrossover': [0,10],
        'PointDensity': [15,6],
        'OverlapPoints': 4,
        'Buffer': 2,
        'I0Points':10,
        'name_of_sample':'Sample'
    }
    DEFAULT_CONSTANTS_TITLES = {
        'menu1': {
            'XPosition': 'Sample X: ',
            'YPosition': 'Sample Y: ',
            'ZPosition': 'Sample Z: ',
            'ZFlipPosition': 'Sample Z Flip: ',
            'DirectBeam': 'Sample Z Direct Beam: ',
        },
        'menu2': {
            'XOffset': 'Sample X Offset: ',
            'ReverseHolder': 'Start at -180 [deg]?',
            'ThetaOffset': 'Sample Theta Offset:'
        },
        'menu3': {
            'IndependentVariable': 'Independent variable: ',
            'DensityCalc': 'Method to calculate point density: ',
            'SampleThickness': 'Approximate thickness [Angstroms]: ',
            'AngleCrossover': 'Angle to change point density: ',
            'PointDensity': 'Point density: ',
            'OverlapPoints': 'Overlap points: ',
            'Buffer': 'Movement buffer points: ',
            'I0Points': 'I0 points: '
        }
    }
    MOTOR_OPTIONS_LAYOUT = widgets.Layout(widget='400px')
    MOTOR_OPTIONS_STYLE = {'description_width': '150px'}
    BUTTON_STYLE = {'button_color':'#63666A', 'text_color':'white'}
    def __init__(self, path="",scriptgen_widget=None, **kwargs):
        # Get default values
        constants = self.DEFAULT_CONSTANTS.copy()
        constants_titles = self.DEFAULT_CONSTANTS_TITLES.copy()
        # Update those based on what the kwargs are -- 
        constants.update({k: v for k, v in kwargs.items() if k in constants})
        constants = clean_for_widgets(constants) # Change the list to a CSV
        # Build the experiment tab
        super().__init__(title='Motor Positions', constants=constants, constants_titles=constants_titles, build_constants=False,scriptgen_widget=scriptgen_widget) # Build the initial stuff

        # Name of Sample Box
        self.name_of_sample = widgets.Text(description="Name of Sample: ", value=constants['name_of_sample'], style=self.MOTOR_OPTIONS_STYLE)
        
        # Initial buttons used to create new measurements (one level down)
        self.add_scan_button = widgets.Button(description="Add new measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        self.duplicate_scan_button = widgets.Button(description="Duplicate measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        self.delete_scan_button = widgets.Button(description="Delete measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        # Add button functionality
        self.add_scan_button.on_click(lambda b: self.add_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.duplicate_scan_button.on_click(lambda b: self.copy_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.delete_scan_button.on_click(lambda b: self.delete_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.control_buttons.children = tuple([self.add_scan_button, self.duplicate_scan_button, self.delete_scan_button])

        accordion_menus = []
        # Recreate the constant values based on the Accordians that organize things better -- 
        for key, menu in constants_titles.items():
            menu_display = []
            for attr, value in menu.items():
                attr_title = attr
                gui_label = value
                attr_value = constants[attr]
                # Cycle through the specifics to create the correct button:
                # CHECK BOX OPTIONS
                if 'ReverseHolder' in attr_title:
                    setattr(
                        self,
                        attr_title,
                        widgets.Checkbox(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                # TEXT OPTIONS THAT ARE COMMA SEPARATED LISTS
                elif 'AngleCrossover' in attr_title or 'PointDensity' in attr_title:
                    setattr(
                        self,
                        attr_title,
                        widgets.Text(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                elif 'DensityCalc' in attr_title or 'IndependentVariable' in attr_title:
                    if 'DensityCalc' in attr_title:
                        attr_options = POINT_DENSITY_METHOD
                    elif 'IndependentVariable' in attr_title:
                        attr_options = X_VARIABLES
                    setattr(
                        self,
                        attr_title,
                        widgets.Dropdown(
                            options=attr_options,
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                else:
                    setattr(
                        self,
                        attr_title,
                        widgets.FloatText(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                menu_display.append(getattr(self, attr_title))
            accordion_menus.append(widgets.VBox(menu_display))
        self.menu_box = widgets.Accordion(
            children=accordion_menus,
            titles=("Motor Positions", "Optional Motor offsets", "More Options")
        )
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
            
        self.GUI_experiment = widgets.VBox([self.name_of_sample, self.widget_title, self.menu_box, self.control_buttons, self.layout])
        
    def update_scan_tab(self, layout, widget_str):   
        widget_list = [getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]
        display = []
        titles = []
        for widget in widget_list:
            display.extend([widget.display()])
            titles.append(f"En: {str(widget.energy_start.value)} [eV]")
            widget.energy_start.observe(self.update_title, names='value')

        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, titles[tab]) 

    def update_title(self, change):
        index = self.layout.selected_index
        scan = getattr(self, f"{self.WIDGET.ALS_NAME}{pad_digits(index+1)}")
        self.layout.set_title(index, f"En: {str(scan.energy_start.value)} [eV]")

class PXR_ScriptGen(ALS_ScriptGenWidget):
    EXPERIMENT_NAME = "PXR"
    WIDGET = PXR_Experiment
    WIDGET_TITLE = "Sample "
    JSON_TITLE = "PXR"
    BUTTON_STYLE = {'button_color':'#5D4777', 'text_color':'white'}

    def __init__(self, path="", **kwargs):
        # Initialize the base widget
        super().__init__(exp_name=self.EXPERIMENT_NAME, child_widget=self.WIDGET, path=path, tab_title=self.WIDGET_TITLE, json_title=self.JSON_TITLE, **kwargs)
        # Buttons involved with adding new samples positions
        self.new_sample_button = widgets.Button(
            description="Add New Sample", layout=self.default_button,
            style=self.BUTTON_STYLE
        )
        self.copy_sample_button = widgets.Button(
            description="Duplicate Sample", layout=self.default_button,
            style=self.BUTTON_STYLE
        )
        self.delete_sample_button = widgets.Button(
            description="Delete Sample", layout=self.default_button,
            style=self.BUTTON_STYLE
        )
        # Add click functionality
        self.new_sample_button.on_click(lambda b: self.new_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.copy_sample_button.on_click(lambda b: self.copy_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.delete_sample_button.on_click(lambda b: self.delete_tab(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))

        self.additional_buttons.children = tuple(
            [
                widgets.HBox([self.new_sample_button, self.copy_sample_button, self.delete_sample_button])
            ]
        )

        self.GenerateExperiment(self.layout, self.WIDGET.ALS_NAME, self.WIDGET, **kwargs)
        display(self.GUI)
        
    def update_experiment_tab(self, layout, widget_str):
        widget_list = [getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]
        display = []
        display_titles = []
        for widget in widget_list:
            display.extend([widget.display()])
            display_titles.extend([widget.name_of_sample.value])
            widget.name_of_sample.observe(self.update_title, names='value')
        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, display_titles[tab])
    
    def update_title(self, change):
        index = self.layout.selected_index
        self.layout.set_title(index, change['owner'].value) # No idea why it is setup this way.
            
    def save_as_df(self):
        df = {}
        # Get the sample dataframe
        df = self.save_as_dict()
        #Temp thing -- 
        MACRO_DIR = str(self.save_dir)

        # Cycle through the samples and start saving data
        for i, (index, sample) in enumerate(df.items()):
            MotorPositions = {}
            VariableMotors = {}
            ScanList = [m for m in sample.keys() if "scan_" in m]
            MotorPositions = {k: v for k, v in sample.items() if k not in ScanList}
            beam_offset_counter = 0
            
            name = self.WIDGET.ALS_NAME+pad_digits(i+1)
            sample_attr = getattr(self, name)
            sample_name = sample_attr.name_of_sample.value
            
            for ii, scan in enumerate(ScanList):
                dm = sample[scan]
                exclude_motors = ['theta_end', 'energy_start', 'energy_stop', 'energy_delta']
                if dm['energy_delta'] == 0:
                    energy = [dm['energy_start']]
                else:
                    energy = np.arange(dm['energy_start'], dm['energy_stop'], dm['energy_delta'])
                #energy = dm['energy']
                theta_end = dm['theta_end']
                #VariableMotors = {k: v for k, v in dm.items() if k not in ['energy', 'theta_end']}
                VariableMotors = {k: v for k, v in dm.items() if k not in exclude_motors}
                for iii, en in enumerate(energy):
                    df_en = pd.DataFrame(AngleRunGenerator_v2(MotorPositions, VariableMotors, en, theta_end))
                    df_en['Sample X'] = df_en['Sample X'] + MotorPositions['XOffset']*beam_offset_counter
                    beam_offset_counter += 1
                    if iii == 0:
                        sampdf = df_en
                    else:
                        sampdf = pd.concat([sampdf, df_en], ignore_index=True)
                #sampdf = pd.DataFrame(AngleRunGenerator_v2(MotorPositions, VariableMotors, energy, theta_end))
                # Some cleanup
                round_this_motor = ['Sample X', 'Sample Y', 'Sample Z', 'Sample Theta', 'CCD Theta']
                # sampdf['Sample X'] = sampdf['Sample X'] + MotorPositions['XOffset']*ii
                for motor in round_this_motor:
                    sampdf[motor] = sampdf[motor].round(4)
                # Initialize RunFile
                if i == 0 and ii == 0:
                    RunFile = sampdf
                else:
                    RunFile = pd.concat([RunFile, sampdf], ignore_index=True)
                if ii == 0:
                    SingleFile = sampdf.copy()
                else:
                    SingleFile = pd.concat([SingleFile, sampdf], ignore_index=True)
            # save the sampdf as a macro file
            SAVE_SCAN = MACRO_DIR + '/' + sample_name +'_PXR_'+str(i) + '.txt'   
            # Make sure the positions are correct
            SingleFile = SingleFile[[c for c in SingleFile if c not in ['Exposure']] + ['Exposure']]
            SingleFile.to_csv(SAVE_SCAN, index=False, sep='\t')
            clean_script(SAVE_SCAN) # Remove exposure and clear last line if it exists
            
                    
        RunFile = RunFile[[c for c in RunFile if c not in ['Exposure']] + ['Exposure']]
        return RunFile 
        
    def save_beamline_scan(self, ext='.txt', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'

        df = self.save_as_dict()
        #df.to_csv(SAVEPATH, index=False, sep=sep)

        
    def clean_scan_file(self, path):
        # Cleanup the output because the ALS requires specific things
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            if not lines: # Empty file
                return
            # Edit first line to remove 'exposure' from headers
            if 'Exposure' in lines[0]:
                lines[0] = lines[0].replace('\tExposure' , '')
            # Edit last line to remove any carriage return that may exist
            if '\n' in lines[-1]:
                lines[-1] = lines[-1].replace('\n', '')

            with open(path, 'w') as f:
                f.writelines(lines)
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
        except Exception as e:
            print(f"An error occured: {e}")
        del lines #Remove it from memory (it can be large)
        
PHOTODIODE_POS = {
    "CCD Y": 100.000,
    "CCD Theta": 0.000,
    "CCD X": 6.000
}

XRR_POS = {
    "CCD Y": 100.000,
    "CCD Theta": 0.000,
    "CCD X": 101.000,
    "Beam Stop": 5.25
}
        
class PXR_MacroExperiment(ALS_ExperimentWidget):
    ALS_NAME = 'exp_'
    WIDGET = PXR_Scan 
    DEFAULT_CONSTANTS = {
        'XPosition': 0.0,
        'YPosition': 0.0,
        #'ZPosition': 0.0,
        #'ZFlipPosition': 0.0,
        #'DirectBeam': -5.0,
        'XOffset': 0.15,
        'ReverseHolder': False,
        #'ThetaOffset': 0.0,
        'IndependentVariable': 'Qval',
        'DensityCalc': 'Use Thickness',
        'SampleThickness': 250,
        'AngleCrossover': [0,10],
        'PointDensity': [15,6],
        'OverlapPoints': 4,
        'Buffer': 2,
        'I0Points':10             
    }
    DEFAULT_CONSTANTS_TITLES = {
        'menu1': {
            'XPosition': 'Sample X: ',
            'YPosition': 'Sample Y: ',
            'XOffset': 'X Offset per scan: ',
            'ReverseHolder': 'Start scan at -180 [deg]?'
        },
        'menu3': {
            'IndependentVariable': 'Independent variable: ',
            'DensityCalc': 'Method to calculate point density: ',
            'SampleThickness': 'Approximate thickness [Angstroms]: ',
            'AngleCrossover': 'Angle to change point density: ',
            'PointDensity': 'Point density: ',
            'OverlapPoints': 'Overlap points: ',
            'Buffer': 'Movement buffer points: ',
            'I0Points': 'I0 points: '
        }
    }
    MOTOR_OPTIONS_LAYOUT = widgets.Layout(widget='400px')
    MOTOR_OPTIONS_STYLE = {'description_width': '150px'}
    BUTTON_STYLE = {'button_color':'#63666A', 'text_color':'white'}
    def __init__(self, path="",scriptgen_widget=None, **kwargs):
        # Get default values
        constants = self.DEFAULT_CONSTANTS.copy()
        constants_titles = self.DEFAULT_CONSTANTS_TITLES.copy()
        # Update those based on what the kwargs are -- 
        constants.update({k: v for k, v in kwargs.items() if k in constants})
        constants = clean_for_widgets(constants) # Change the list to a CSV
        # Build the experiment tab
        super().__init__(title='Motor Positions', constants=constants, constants_titles=constants_titles, build_constants=False,scriptgen_widget=scriptgen_widget) # Build the initial stuff

        # Name of Sample Box
        self.name_of_sample = widgets.Text(description="Name of Sample: ", value="Sample", style=self.MOTOR_OPTIONS_STYLE)
        
        # Initial buttons used to create new measurements (one level down)
        self.add_scan_button = widgets.Button(description="Add new measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        self.duplicate_scan_button = widgets.Button(description="Duplicate measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        self.delete_scan_button = widgets.Button(description="Delete measurement", layout=self.default_button, style=self.BUTTON_STYLE)
        # Add button functionality
        self.add_scan_button.on_click(lambda b: self.add_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.duplicate_scan_button.on_click(lambda b: self.copy_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.delete_scan_button.on_click(lambda b: self.delete_scan(self.layout, self.WIDGET, self.WIDGET.ALS_NAME))
        self.control_buttons.children = tuple([self.add_scan_button, self.duplicate_scan_button, self.delete_scan_button])

        accordion_menus = []
        # Recreate the constant values based on the Accordians that organize things better -- 
        for key, menu in constants_titles.items():
            menu_display = []
            for attr, value in menu.items():
                attr_title = attr
                gui_label = value
                attr_value = constants[attr]
                # Cycle through the specifics to create the correct button:
                # CHECK BOX OPTIONS
                if 'ReverseHolder' in attr_title:
                    setattr(
                        self,
                        attr_title,
                        widgets.Checkbox(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                # TEXT OPTIONS THAT ARE COMMA SEPARATED LISTS
                elif 'AngleCrossover' in attr_title or 'PointDensity' in attr_title:
                    setattr(
                        self,
                        attr_title,
                        widgets.Text(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                elif 'DensityCalc' in attr_title or 'IndependentVariable' in attr_title:
                    if 'DensityCalc' in attr_title:
                        attr_options = POINT_DENSITY_METHOD
                    elif 'IndependentVariable' in attr_title:
                        attr_options = X_VARIABLES
                    setattr(
                        self,
                        attr_title,
                        widgets.Dropdown(
                            options=attr_options,
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                else:
                    setattr(
                        self,
                        attr_title,
                        widgets.FloatText(
                            value=attr_value,
                            description=gui_label,
                            layout=self.MOTOR_OPTIONS_LAYOUT,
                            style=self.MOTOR_OPTIONS_STYLE
                        )
                    )
                menu_display.append(getattr(self, attr_title))
            accordion_menus.append(widgets.VBox(menu_display))
        self.menu_box = widgets.Accordion(
            children=accordion_menus,
            titles=("Motor Positions", "Scan Options")
        )
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
            
        self.GUI_experiment = widgets.VBox([self.name_of_sample, self.widget_title, self.menu_box, self.control_buttons, self.layout])
        
    def update_scan_tab(self, layout, widget_str):   
        widget_list = [getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]
        display = []
        titles = []
        for widget in widget_list:
            display.extend([widget.display()])
            titles.append(f"En: {str(widget.energy_start.value)} [eV]")
            widget.energy_start.observe(self.update_title, names='value')

        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, titles[tab]) 

    def update_title(self, change):
        index = self.layout.selected_index
        scan = getattr(self, f"{self.WIDGET.ALS_NAME}{pad_digits(index+1)}")
        self.layout.set_title(index, f"En: {str(scan.energy_start.value)} [eV]")
        
class PXR_MacroGen(PXR_ScriptGen):
    WIDGET = PXR_MacroExperiment

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.GUI.children = () # temporary fix to clear the widget generated by PXR_ScriptGen
        # New banner
        self.title_banner = widgets.HTML(value=f"<h2>{self.EXPERIMENT_NAME} Macro Generator</h2>") 

        # Change the button
        self.save_button_beamline = widgets.Button(
            description = "Save Beamline Scan File",
            layout=self.default_button,
            style=self.banner_button_style
            )
        self.save_button_beamline.on_click(self.save_beamline_scan)
        self.save_buttons = widgets.HBox([self.save_button_beamline])
        self.GUI = widgets.VBox([self.title_banner, self.control_buttons, self.layout, self.json_buttons,self.save_button_beamline], layout=self.default_GUI)      
        display(self.GUI)

        # Everything is the same -- just save as a macro instead
    
    def save_beamline_scan(self, ext='.txt', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'
        
        df = self.save_as_dict()
        macro_details = self._build_individual_scan_files(df)
        
                
        macro = begin_macro(SAVENAME) # list of commands that will become the spiral scan
        # Move the sample plate into the 'zero' position to begin alignment -- 
        macro += [move_motor('Sample Theta', 0.0, 0.1)]
        macro += [move_motor('Sample Z', 0, 0.1)]
        
        # Run add an I0 here --- 
        
        # Begin samples
        for sample in macro_details:
            name = sample['name']
            posx = sample['posx']
            posy = sample['posy']
            path = sample['path']
            
            # Each sample -- 
            macro += [add_comment(f"Begin Sample: {name}")]
            macro += [clear_instruments]
            macro.extend(run_dict_trajectory(PHOTODIODE_POS))
            macro += [move_motor('Sample X', posx, 0.1)]
            macro += [move_motor('Sample Y', posy, 0.1)]
            macro.extend(XRR_lineup(repeat=3))
            macro.extend(run_dict_trajectory(XRR_POS))
            macro += [set_instrument("CCD", exp=0.0, extension='fits')]
            macro += [add_comment(f"Add from file here: {name}")]
            # macro += [analog_from_file(name, path)]
        macro += [clear_instruments]
        macro.extend(run_dict_trajectory(PHOTODIODE_POS))
        macro += [add_prompt(f"Finished Macro")]
        build_macro(SAVENAME, SAVEDIR, macro)
    
    
    def _build_individual_scan_files(self, df):
        # Initialize the macro construction        
        all_sample_scripts = [] # Contains lists associated with each detector position
        
        MACRO_DIR = str(self.save_dir)
    
        # Cycle through samples and start saving data
        for i, (index, sample) in enumerate(df.items()):
            # Get the name of the sample as chosen by user
            name = self.WIDGET.ALS_NAME+pad_digits(i+1)
            sample_attr = getattr(self, name)
            sample_name = sample_attr.name_of_sample.value
            
            # Get Motor positions for XRR
            MotorPositions = {}
            VariableMotors = {}
            ScanList = [m for m in sample.keys() if "scan_" in m]
            MotorPositions = {k: v for k, v in sample.items() if k not in ScanList}
            beam_offset_counter = 0
            
            # Update Motor Positions that will be overwritten by macro generation
            MotorPositions['ZPosition'] = 0.00
            MotorPositions['ZFlipPosition'] = 3.1435
            MotorPositions['DirectBeam'] = -2.00
            MotorPositions['ThetaOffset'] = 0.00
            
            for ii, scan in enumerate(ScanList):
                dm = sample[scan]
                exclude_motors = ['theta_end', 'energy_start', 'energy_stop', 'energy_delta']
                if dm['energy_delta'] == 0:
                    energy = [dm['energy_start']]
                else:
                    energy = np.arange(dm['energy_start'], dm['energy_stop'], dm['energy_delta'])
                theta_end = dm['theta_end']
                VariableMotors = {k: v for k, v in dm.items() if k not in exclude_motors}
                for iii, en in enumerate(energy):
                    df_en = pd.DataFrame(AngleRunGenerator_v2(MotorPositions, VariableMotors, en, theta_end))
                    df_en['Sample X'] = df_en['Sample X'] + MotorPositions['XOffset']*beam_offset_counter
                    beam_offset_counter += 1
                    if iii == 0:
                        sampdf = df_en
                    else:
                        sampdf = pd.concat([sampdf, df_en], ignore_index=True)
                # Some cleanup
                round_this_motor = ['Sample X', 'Sample Y', 'Sample Z', 'Sample Theta', 'CCD Theta']
                for motor in round_this_motor:
                    sampdf[motor] = sampdf[motor].round(4)
                    
                # Initialize RunFile
                if ii == 0:
                    RunFile = sampdf
                else:
                    RunFile = pd.concat([RunFile, sampdf], ignore_index=True)
                   
            # save the sampdf as a macro file
            SAVE_SCAN = MACRO_DIR + '/' + sample_name +'_PXR_'+str(i) + '.txt'   
            # Make sure the positions are correct
            RunFile = RunFile[[c for c in RunFile if c not in ['Exposure']] + ['Exposure']]
            
            RunFile.to_csv(SAVE_SCAN, index=False, sep='\t')
            clean_script(SAVE_SCAN) # Remove exposure and clear last line if it exists
            # Save all details required for the macro
            macro_stats = {}
            macro_stats['name'] = sample_name
            macro_stats['posx'] = RunFile['Sample X'].iloc[0]
            macro_stats['posy'] = RunFile['Sample Y'].iloc[0]
            macro_stats['path'] = SAVE_SCAN
            
            all_sample_scripts.append(macro_stats) # Save path for macro generation
            
        return all_sample_scripts

def AngleRunGenerator_v2(MotorPositions, VariableMotors, Energy, theta_end):
    #Constants ## https://www.nist.gov/si-redefinition
    SOL = 299792458 #m/s
    PLANCK_JOULE = 6.6267015e-34 #Joule s
    ELEMCHARGE =  1.602176634e-19 #coulombs
    PLANCK = PLANCK_JOULE / ELEMCHARGE #eV s
    meterToAng = 10**(10)
    ##Initialization of needed components
    Wavelength = SOL * PLANCK * meterToAng / Energy
    # Motor Positions for the Sample level
    XPosition = MotorPositions['XPosition']
    YPosition = MotorPositions['YPosition']
    ZPosition = MotorPositions['ZPosition']
    Z180Position = MotorPositions['ZFlipPosition']
    Zdelta = ZPosition - Z180Position
    ThetaOffset = MotorPositions['ThetaOffset']
    # Point Density parameters
    AngleCrossover = MotorPositions['AngleCrossover']
    PointDensity = MotorPositions['PointDensity']
    DensityCalc = MotorPositions['DensityCalc']
    IndependentVariable = MotorPositions['IndependentVariable']
    SampleThickness = MotorPositions['SampleThickness']
    OverlapPoints = int(MotorPositions['OverlapPoints'])
    # From the Measurement Level
    AngleChange = VariableMotors['Sample Theta'].copy()
    AngleChange.append(theta_end)
    AngleNumber = len(VariableMotors['Sample Theta'])
    # Begin cycling through the angle points
    for i in range(AngleNumber):
        if i == 0: # Run initialization
            ##Calculate the start and stop location for Q
            AngleStart = AngleChange[i] # All of the relevant values are in terms of angles, but Q is calculated as a check
            AngleStop = AngleChange[i+1]
            QStart = 4*np.pi*np.sin(np.radians(AngleStart))/Wavelength
            QStop = 4*np.pi*np.sin(np.radians(AngleStop))/Wavelength

            # Calculate the point density based on where this cycle ends
            AngleDensity = locate_point_density(AngleStop, AngleCrossover, PointDensity)

            # Calculate the dq/dtheta as requested
            if DensityCalc == 'Use Thickness':
                dq = 2*np.pi/(SampleThickness*AngleDensity) # Use thickness to approximate fringe size
                Points = int(np.ceil((QStop - QStart)/dq))
            elif DensityCalc == 'Use Delta':
                if IndependentVariable == 'Qval':
                    dq = AngleDensity # Give an absolute points density in dq
                    Points = int(np.ceil((QStop - QStart)/dq))
                elif IndependentVariable == 'Theta':
                    dtheta = AngleDensity # Give an absolute point density in dtheta
                    Points = int(np.ceil((AngleStop-AngleStart)/dtheta))
            QList = np.linspace(QStart, QStop, Points).tolist() # Initialize the QList based on initial configuration
            SampleTheta = np.linspace(AngleStart, AngleStop, Points)# Begin generating list of 'Sample Theta' locations to take data
            CCDTheta = SampleTheta*2 # Make corresponding CCDTheta positions
            SampleTheta = SampleTheta+ThetaOffset # If running multiple samples in a row this will offset the sample theta based on alignment ##CCD THETA SHOULD NOT BE CHANGED

            # Create lists for SampleX/Y/Z at the correct length
            SampleX=[XPosition]*len(QList)
            BeamLineEnergy=[Energy]*len(QList)
            SampleY = YPosition+Zdelta/2+Zdelta/2*np.sin(np.radians(SampleTheta)) #Adjust 'Sample Y' based on the relative axis of rotation
            SampleZ = ZPosition+Zdelta/2*(np.cos(np.radians(SampleTheta))-1) #Adjust 'Sample Z' based on the relative axis of rotation

            #Convert numpy arrays into lists for Pandas generation
            SampleTheta=SampleTheta.tolist()
            SampleY=SampleY.tolist()
            SampleZ=SampleZ.tolist()
            CCDTheta=CCDTheta.tolist()

            # Generate all other motors based on the input Variable motors
            Variable_Motor_dict = {}
            for key, value in VariableMotors.items(): # Cycle through the Variable Motor dictionary
                if 'Theta' not in key:
                    Variable_Motor_dict[key] = []
                    Variable_Motor_dict[key].extend([VariableMotors[key][i]]*len(QList))

            # Adding one point at the start to measure the direct beam cut in half
            #QList.insert(0,0)
            #SampleTheta.insert(0,0)
            #CCDTheta.insert(0,0)
            #SampleX.insert(0,SampleX[0])
            #SampleY.insert(0,YPosition)
            #SampleZ.insert(0,ZPosition) # Should be cutting the beam in half at this position
            #BeamLineEnergy.insert(0,BeamLineEnergy[0])
            # All Variable Motors
            #for motor in Variable_Motor_dict:
            #    Variable_Motor_dict[motor].insert(0, Variable_Motor_dict[motor][0])
                
            # Adding Additional points at the start to assess error in beam intensity
            for d in range(int(MotorPositions['I0Points'])):
                QList.insert(0,0)
                #ThetaInsert = 0 if MotorPositions['ReverseHolder']==0 else -180
                SampleTheta.insert(0,0)
                CCDTheta.insert(0,0)
                SampleX.insert(0,SampleX[0])
                SampleY.insert(0,YPosition)
                SampleZ.insert(0,MotorPositions['DirectBeam'])
                BeamLineEnergy.insert(0,BeamLineEnergy[0])
                # All Variable Motors
                for motor in Variable_Motor_dict:
                    Variable_Motor_dict[motor].insert(0, Variable_Motor_dict[motor][0])
        
        else: # for all of the ranges after the first set of samples
            ##Section is identical to the above
            AngleStart = AngleChange[i] # All of the relevant values are in terms of angles, but Q is calculated as a check
            try:
                AngleStop = AngleChange[i+1]
            except:
                AngleStop = theta_end

            QStart = 4*np.pi*np.sin(np.radians(AngleStart))/Wavelength
            QStop = 4*np.pi*np.sin(np.radians(AngleStop))/Wavelength

            # Calculate the point density based on where this cycle ends.
            AngleDensity = locate_point_density(AngleStop, AngleCrossover, PointDensity)
            # Calculate the dq/dtheta as requested
            if DensityCalc == 'Use Thickness':
                dq = 2*np.pi/(SampleThickness*AngleDensity) # Use thickness to approximate fringe size
                Points = int(np.ceil((QStop - QStart)/dq))
            elif DensityCalc == 'Use Delta':
                if IndependentVariable == 'Qval':
                    dq = AngleDensity # Give an absolute points density in dq
                    Points = int(np.ceil((QStop - QStart)/dq))
                elif IndependentVariable == 'Theta':
                    dtheta = AngleDensity # Give an absolute point density in dtheta
                    Points = int(np.ceil((AngleStop-AngleStart)/dtheta))
            QListAddition = np.linspace(QStart, QStop, Points).tolist() # Initialize the QList based on initial configuration
            SampleThetaAddition = np.linspace(AngleStart, AngleStop, Points).tolist() # Begin generating list of 'Sample Theta' locations to take data
            for p in range(OverlapPoints):
                QListAddition.insert(0, QList[-1*(p+2)])
                SampleThetaAddition.insert(0, SampleTheta[-1*(p+2)]-ThetaOffset)
            SampleThetaAdditionArray=np.asarray(SampleThetaAddition) #Convert back to numpy array

            CCDThetaAddition=SampleThetaAdditionArray*2 #Calculate the CCD theta POsitions
            CCDThetaAddition=CCDThetaAddition.tolist() #Convert to list
            SampleThetaAdditionArray=SampleThetaAdditionArray+ThetaOffset #Account for theta offset
            SampleThetaAddition = SampleThetaAdditionArray.tolist()

            SampleXAddition=[XPosition]*len(QListAddition)
            BeamLineEnergyAddition=[Energy]*len(QListAddition)
            
            SampleYAddition=YPosition+Zdelta/2+Zdelta/2*np.sin(np.radians(SampleThetaAdditionArray))
            SampleZAddition=ZPosition+Zdelta/2*(np.cos(np.radians(SampleThetaAdditionArray))-1) 
            SampleYAddition=SampleYAddition.tolist()
            SampleZAddition=SampleZAddition.tolist()

            # Generate all other motors based on the input Variable motors
            Variable_Motor_dict_Addition = {}
            for key, value in VariableMotors.items(): # Cycle through the Variable Motor dictionary
                if 'Theta' not in key:
                    Variable_Motor_dict_Addition[key] = []
                    Variable_Motor_dict_Addition[key].extend([VariableMotors[key][i]]*len(QListAddition))
                    
            #Check to see if any of the variable motors have moved to add buffer points
            for key, value in VariableMotors.items():
                if 'Theta' not in key:
                    if VariableMotors[key][i] != VariableMotors[key][i-1]:
                        for d in range(int(MotorPositions['Buffer'])):
                            QListAddition.insert(0,QListAddition[d])
                            SampleThetaAddition.insert(0,SampleThetaAddition[d])
                            CCDThetaAddition.insert(0,CCDThetaAddition[d])
                            SampleXAddition.insert(0,SampleXAddition[d])
                            SampleYAddition.insert(0,SampleYAddition[d])
                            SampleZAddition.insert(0,SampleZAddition[d])
                            BeamLineEnergyAddition.insert(0,BeamLineEnergyAddition[d])
                            for motor in Variable_Motor_dict_Addition:
                                Variable_Motor_dict_Addition[motor].insert(0, Variable_Motor_dict_Addition[motor][d])
            
            QList.extend(QListAddition)
            SampleTheta.extend(SampleThetaAddition)
            CCDTheta.extend(CCDThetaAddition)
            SampleX.extend(SampleXAddition)
            SampleY.extend(SampleYAddition)
            SampleZ.extend(SampleZAddition)
            BeamLineEnergy.extend(BeamLineEnergyAddition)
            for motor, addition in zip(Variable_Motor_dict.keys(), Variable_Motor_dict_Addition.keys()):
                Variable_Motor_dict[motor].extend(Variable_Motor_dict_Addition[addition])
        
        #Check what side the sample is on. If on the bottom, sample theta starts @ -180
    if MotorPositions['ReverseHolder'] == 1:
        SampleTheta=[theta-180 for theta in SampleTheta] # for samples on the backside of the holder, need to check and see if this is correct

    sampdf = {}
    sampdf['Sample X'] = SampleX
    sampdf['Sample Y'] = SampleY
    sampdf['Sample Z'] = SampleZ
    sampdf['Sample Theta'] = SampleTheta
    sampdf['CCD Theta'] = CCDTheta
    sampdf['Beamline Energy'] = BeamLineEnergy
    for key, item in Variable_Motor_dict.items():
        sampdf[key] = item
   
    return sampdf#(SampleX, SampleY, SampleZ, SampleTheta, CCDTheta, HOSList, HESList, BeamLineEnergy, ExposureList, QList)


    
    








def locate_point_density(angle, crossover_list, density_list):
    for i in range(len(crossover_list)-1):
        if int(crossover_list[i]) <= angle <= int(crossover_list[i+1]):
            return int(density_list[i])
    return int(density_list[-1])
    
def update_dict(old_dict, new_dict):
    for key in old_dict.keys():
        if key in new_dict:
            old_dict[key] = new_dict[key]
    return old_dict
    
def clean_for_widgets(my_dict):
    for key, value in my_dict.items():
        if isinstance(value, list):
            my_dict[key] = ', '.join(map(str, value))
    return my_dict
