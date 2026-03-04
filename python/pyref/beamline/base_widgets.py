"""Base widgets for ALS beamline script generation.

This module provides base classes for creating widgets that generate run scripts
for ALS beamline 11.0.1.2. It includes functionality for saving, loading,
verifying, and managing experiment configurations.

Requirements:
    numpy
    pandas
    ipywidgets
    PyQt6
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

class ALS_ScriptGenWidget:
    """
    Base class for ALS script generation widgets.

    This class provides basic functions for creating widgets that generate
    run scripts. It includes functionality for saving,
    loading, verifying, and managing experiment configurations.
    
    """
    
    default_button = widgets.Layout(width='200px')
    default_GUI = widgets.Layout(widget='1200px')
    banner_button_style = {'button_color':'#007681', 'text_color':'white'}
    
    
    def __init__(self, exp_name="", child_widget=None, path=None, tab_title=None, json_title=None, **kwargs):
        """
        Initializes an ALS_ScriptGenWidget. This is a base class to create widgets specializing in run-script generation
        """

        self.child_widget = child_widget if child_widget is not None else ALS_ExperimentWidget
        self._save_dir = path if path is not None else ""
        self.tab_title = tab_title if tab_title is not None else "Experiment "
        self.json_title = json_title if json_title is not None else "Default"

        # Create a title banner for the full widget
        self.title_banner = widgets.HTML(value=f"<h2>{exp_name} Script Generator</h2>") 

        # Default save / path generation
        # Name that will be used to save the
        self.save_name = widgets.Text(
            value="",
            description = "Name of Script: ",
            layout=widgets.Layout(width='300px'),
            style = {'description_width': '90px'}
        )

        self.save_button_CCD = widgets.Button(
            description = "Save CCD Scan File",
            layout=self.default_button,
            style = self.banner_button_style
        )
        self.save_button_CCD.on_click(self.save_script)
        
        self.save_button_beamline = widgets.Button(
            description = "Save Beamline Scan File",
            layout=self.default_button,
            style = self.banner_button_style
        )
        self.save_button_beamline.on_click(self.save_script)
        self.save_buttons = widgets.HBox([self.save_button_CCD])#, self.save_button_beamline])

        # Browse directory functions
        self.browse_directory_button = widgets.Button(
            description="Choose Save Directory", layout=self.default_button,
            style=self.banner_button_style
        )
        self.browse_directory_button.on_click(self.browse_save_directory)
        # Display Save Directory
        self.display_save_directory = widgets.HTML(
            value = "<b>Save Directory:</b>" + str(self._save_dir),
            description = ""
        )

        # Additional buttons that can be added in children classes
        self.additional_buttons = widgets.VBox([]) # Empty for now

        # Create the widget that will contain all buttons:
        self.control_buttons = widgets.VBox(
            [
                self.save_name,
                widgets.HBox([self.browse_directory_button, self.display_save_directory]),
                self.additional_buttons
            ]
        )
        
        #self.control_buttons = widgets.VBox(
        #    [
        #        self.save_name,
        #        #widgets.HBox([self.browse_directory_button, self.verify_button, self.save_button]),
        #        widgets.HBox([self.browse_directory_button, self.save_button]),
        #        self.display_save_directory,
        #        self.additional_buttons
        #    ]
        #)

        # Build the layout/GUI based on the experiment requirements
        self.layout = widgets.Tab(children=[], layout=self.default_GUI) # Default layout that will contain information

        # Save button to JSON
        self.json_button_style = {'button_color':'#672E45', 'text_color':'white'}
        self.save_json_button = widgets.Button(
            description = "Save Script to JSON",
            layout=self.default_button,
            style = self.json_button_style
        )
        self.load_json_button = widgets.Button(
            description = "Load Script from JSON",
            layout=self.default_button,
            style = self.json_button_style
        )
        self.load_json_button.on_click(self.load_json)
        self.save_json_button.on_click(self.save_json)
        self.json_buttons = widgets.HBox([self.save_json_button, self.load_json_button])

        self.GUI = widgets.VBox([self.title_banner, self.control_buttons, self.layout, self.json_buttons,self.save_buttons], layout=self.default_GUI)        
        #display(self.GUI)

    def __call__(self):
        return self.save_as_dict()

    def __len__(self):
        return 0
    

    @property
    def save_dir(self):
        return self._save_dir
    
    @save_dir.setter
    def save_dir(self, val):
        self._save_dir=val
        self.display_save_directory.value = "<b>Save Directory: </b>" + str(self._save_dir)

    def GenerateExperiment(self, layout, widget_str, widget, **kwargs):
        # Create the samples
        for key, value in kwargs.items():
            if widget_str in key:
                df = getattr(value, "output_dict", lambda: value)()
                self.new_tab(layout, widget, widget_str, **df)
                
        if not layout.children:
            df = {}
            self.new_tab(layout, widget, widget_str, **df)
            
        return layout

    def new_tab(self, layout, widget, widget_str, b=None,  **kwargs):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]) + 1 # Next Index
        try:
            new_widget = widget(scriptgen_widget=self, **kwargs)
            setattr(self, widget_str + pad_digits(index), new_widget)
            self.update_experiment_tab(layout, widget_str)
        except Exception as e:
            print(f"Error creating new tab: {e}")
            import traceback
            traceback.print_exc()
            raise

    def copy_tab(self, layout, widget, widget_str, b=None):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]) + 1  # Last Index
        # Get stats of copied obj
        open_tab = self.layout.selected_index + 1
        target_widget_name = widget_str + pad_digits(open_tab)
        if hasattr(self, target_widget_name):
            df = getattr(self, widget_str + pad_digits(open_tab)).output_dict()
        else:
            df = {}
        setattr(self, widget_str + pad_digits(index), widget(scriptgen_widget=self, **df))
        self.update_experiment_tab(layout, widget_str)
        
    def delete_tab(self, layout, widget, widget_str, b=None):
        try:
            index = self.layout.selected_index + 1
        except:
            index = 0
        widget_name = widget_str+pad_digits(index)
        if hasattr(self, widget_name):
            delattr(self, widget_name)
            
            j = index + 1
            while hasattr(self, f"{widget_str}{pad_digits(j)}"):
                setattr(self, f"{widget_str}{pad_digits(j-1)}", getattr(self, f"{widget_str}{pad_digits(j)}"))
                delattr(self, f"{widget_str}{pad_digits(j)}")
                j += 1
            self.update_experiment_tab(layout, widget_str)
            
    def clean_slate(self):
        self.layout.children = ()
        for key in (key for key in dir(self) if not key.startswith("__")):
            attr = getattr(self, key)
            if not inspect.isclass(attr) and hasattr(attr, 'ALS_NAME'):
                delattr(self, key)

    def update_experiment_tab(self, layout, widget_str):
        widget_list = [getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]
        display = []
        for widget in widget_list:
            display.extend([widget.display()])
        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, self.tab_title +str(tab+1)) 
            
    def save_script(self, ext='.txt', sep='\t', b=None):
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.txt'

        df = self.save_as_df()
        df.to_csv(SAVEPATH, index=False, sep=sep)

        # Cleanup the output because the ALS requires specific things
        try:
            with open(SAVEPATH, 'r') as f:
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

    def verify_script(self, b=None):
        return 0

    def save_as_df(self):
        df = self.save_as_dict()
        return pd.DataFrame(df, ignore_index=True)

    def save_as_dict(self):
        save_dict = {} # output dictionary
        #level = 0 # initial level
        for key in (key for key in dir(self) if not key.startswith("__")):
            attr = getattr(self, key)
            #if not hasattr(attr, 'WIDGET'):
            #    continue # Pass everything except a potential ALS_UTILITIY
            #if not hasattr(attr.WIDGET, 'ALS_NAME'):
            #    continue
                
            #print(attr)
            #save_dict[key] = self._process_attr_dict(attr) 
                
            if hasattr(attr, 'output_dict') and not inspect.isclass(attr): #not inspect.isclass(attr) and hasattr(attr, 'WIDGET') and
                save_dict[key] = self._process_attr_dict(self, attr)
        return save_dict

    def _process_attr_dict(self, old_attr, attr): # Recursive helper class to cycle through attributes
        my_dict = attr.output_dict() # This creates the next level dictionary
        for key in (key for key in dir(attr) if not key.startswith("__")):#dir(attr):
            next_attr = getattr(attr, key)
            if type(old_attr) == type(next_attr): # Used to make sure it does not do an infite recursion by going back into itself.
                continue
            if hasattr(next_attr, 'output_dict') and not inspect.isclass(next_attr):
                my_dict[key] = self._process_attr_dict(attr, next_attr)

            #if not inspect.isclass(next_attr) and hasattr(next_attr, 'WIDGET') and hasattr(next_attr, 'output_dict'):
                #my_dict[key] = self._process_attr_dict(next_attr)
        return my_dict

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
                try:
                    root.quit()
                except:
                    pass
                root = None
                gc.collect()
        
        if selected_folder:
            return Path(selected_folder)
        return None

    def save_json(self, b=None):
        df = self.save_as_dict()
        SAVEDIR = str(self.save_dir)
        SAVENAME = str(self.save_name.value)
        SAVEPATH = SAVEDIR + '/' + SAVENAME + '.json'
            
        try:
            with open(SAVEPATH, 'w') as f:
                f.write(f"# {self.json_title}\n")
                json.dump(df, f, indent=4)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error encoding to JSON: {e}")

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
                try:
                    root.quit()
                except:
                    pass
                root = None
                gc.collect()
        
        if selected_file:
            try:
                with open(selected_file, 'r') as f:
                    title_line = f.readline().strip()
                    if title_line == f"# {self.json_title}":
                        df = json.load(f)
                    else:
                        print(f"Not an appropriate script. Expected title: {self.json_title}, got: {title_line}")
                        return None
                self.clean_slate()
                self.GenerateExperiment(self.layout, self.child_widget.ALS_NAME, self.child_widget, **df)
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                import traceback
                traceback.print_exc()
                return None
        return None
            




class ALS_ExperimentWidget:
    # Default styling for the Experiment level
    default_button = widgets.Layout(width='200px')
    default_button_style = {'button_color':'#00313C', 'text_color':'white'}
    default_FloatText = widgets.Layout(width='400px')
    default_FloatText_style = {'description_width': '200px'}
    default_GUI = widgets.Layout(widget='1200px')
    def __init__(self, title="Generic Experiment", constants=None, constants_titles=None, build_constants=True, scriptgen_widget = None):
        self.ScriptgenWidget = scriptgen_widget
        # Initial conditions
        self.constants = constants if constants is not None else {}
        self.constants_titles = constants_titles if constants_titles is not None else {}
        
        # Title of the widget
        self.widget_title = widgets.HTML(value=f"<b>{title}</b>")

        # Create a series of float texts that cover the 'constant values'
        # This can be overridden if the requirement is more complex (ie. Reflectivity)
        if build_constants:
            temp_display = []
            for key, val in self.constants_titles.items():
                attr_title = key
                gui_label = val
                attr_value = self.constants[key]
                if isinstance(attr_value, float):
                    setattr(
                        self,
                        attr_title,
                        widgets.FloatText(
                            description = gui_label,
                            value = attr_value,
                            layout = self.default_FloatText,
                            style = self.default_FloatText_style
                        )
                    )  
                    temp_display.append(getattr(self, attr_title))

                elif isinstance(attr_value, str): # THIS WILL NOT SAVE THIS VALUE. STRINGS SHOULD NOT BE PASSED
                    setattr(
                        self,
                        attr_title,
                        widgets.Text(
                            description = gui_label,
                            value = attr_value,
                            layout = self.default_FloatText,
                            style = self.default_FloatText_style
                        )
                    )   
            self.menu_box = widgets.VBox(temp_display)
        else:
            self.menu_box = widgets.VBox([])
        self.control_buttons = widgets.HBox([]) # Empty Controls
        self.layout = widgets.Tab(children=[], layout=self.default_GUI)
        self.GUI_experiment = widgets.VBox([self.widget_title, self.menu_box, self.control_buttons, self.layout])

    def display(self):
        return self.GUI_experiment

    def add_scan(self, layout, widget, widget_str, b=None, **kwargs):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)])+1 # Next index
        setattr(self, widget_str + pad_digits(index), widget(experiment_widget=self, **kwargs))
        self.update_scan_tab(layout, widget_str)

    def copy_scan(self, layout, widget, widget_str, b=None):
        index = len([getattr(self, attr) for attr in dir(self) if attr.startswith(widget_str)]) + 1  # Last Index
        # Get stats of copied obj
        open_tab = self.layout.selected_index + 1
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
            index = 0
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
        for widget in widget_list:
            display.extend([widget.display()])
        self.layout.children = display
        for tab in np.arange(len(layout.children)):
            layout.set_title(tab, widget_str +str(tab+1)) 

    def output_dict(self):
        df = {}
        for key in self.constants:
            attr = getattr(self, key)
            val = attr.value
            if type(val) is str and self.comma_separated_list(val):
                df[key] = [item.strip() for item in val.split(',')]
            else:
                df[key] = val
        for key in (key for key in dir(self) if not key.startswith("__")):
            attr = getattr(self, key)
            if isinstance(attr, object) and hasattr(attr, 'ALS_NAME') and "WIDGET" not in key:
                df[key] = attr
        return df
    def comma_separated_list(self, input_str):
        pattern = r"^([-+]?\d*\.?\d+)(,\s*[-+]?\d*\.?\d+)*$" # magic from AI
        match = re.match(pattern, input_str)
        return bool(match)

        
        
class ALS_MeasurementWidget:
    LABEL_SIZE = widgets.Layout(width='150px')
    LABEL_STYLE = {'description_width': '150px'}
    CELL_SIZE = widgets.Layout(width='100px')

    def __init__(self, constant_motor_title="Generic Measurement", update_options={}, experiment_widget=None):
        # Set values and update if parameters are loaded -- 
        self.constant_motor_title = widgets.HTML(value=f"<b>{constant_motor_title}</b>") # All tables have a title
        self.constant_motor_attrs = [] # All tables will have some fixed values
        self.table = widgets.VBox(children=[])
        self.control_buttons = widgets.VBox(children=[])
        
        self.ExperimentWidget = experiment_widget
        
    def build_display_table(self):
        build_table = []
        build_table.extend([self.constant_motor_title]) # The title of fixed values
        build_table.extend([getattr(self, widget) for widget in self.constant_motor_attrs]) # Load in the values
        build_table.extend([self.table])  # table object if needed
        build_table.extend([self.control_buttons]) # optional buttons
        return widgets.VBox(build_table)
        
    def display(self):
        return self.display_table

    def get_table(self, include_title=False):
        # Returns the NAMES of attributes that are either fixed or variable
        fixed_content = [attr for attr in dir(self) if attr.startswith("fixed_")]
        variable_content = [attr for attr in dir(self) if attr.startswith("variable_")]
        if include_title:
            fixed_content.insert(0, 'table_titles')
        return fixed_content + variable_content

    def update_table(self):
        if not hasattr(self, 'table_titles') or not self.table_titles:
            return []
        
        if isinstance(self.table_titles[0], list):
            title_row = self.table_titles
        else:
            title_row = [self.table_titles]
        
        fixed_content = [getattr(self, attr) for attr in dir(self) if attr.startswith('fixed_')]
        variable_content = [attr for attr in dir(self) if attr.startswith('variable_')]
        variable_content = self._sort_attrs(variable_content)
        variable_content = [getattr(self, attr) for attr in variable_content]
        
        rows = title_row + fixed_content + variable_content
        output_table = []
        for row in rows:
            if isinstance(row, list) and len(row) > 0:
                if isinstance(row[0], list):
                    for sub_row in row:
                        if isinstance(sub_row, list) and len(sub_row) > 0:
                            output_table.append(widgets.HBox(sub_row))
                else:
                    output_table.append(widgets.HBox(row))
        return output_table
        
    def _sort_attrs(self, attrs):
        def extract_number(s):
            return int(re.search(r'_(\d+)', s).group(1)) #finds the number after the underscore
        return sorted(attrs, key=extract_number)
        

    def output_dict(self):
        df = {}
        for val in self.constant_motor_attrs:
            df[val] = getattr(self, val).value
        for motor in self.get_table():
            item = getattr(self, motor)
            for row in item:
                for i, col in enumerate(row):
                    if i==0:
                        key=col.value
                        df[key] = []
                    else:
                        df[key] += [col.value]
        return df
        
        
        
        
        
        
"""
other functions
"""
def clean_script(file):
    # Cleanup the output because the ALS requires specific things
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        if not lines: # Empty file
            return
        # Edit first line to remove 'exposure' from headers
        if 'Exposure' in lines[0]:
            lines[0] = lines[0].replace('\tExposure' , '')
        # Edit last line to remove any carriage return that may exist
        if '\n' in lines[-1]:
            lines[-1] = lines[-1].replace('\n', '')

        with open(file, 'w') as f:
            f.writelines(lines)
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
    except Exception as e:
        print(f"An error occured: {e}")
    del lines #Remove it from memory (it can be large)
    
    
    
def pad_digits(number):
    """Pads a float or string representation of a float with leading zeros
    until it becomes a 4-digit number (before the decimal point).

    Args:
    number: A float or a string representing a float.

    Returns:
    A string representing the number padded with leading zeros to 4 digits
    before the decimal point. If the integer part already has 4 or more
    digits, it is returned as is (with any decimal part).
    Returns None if the input cannot be converted to a float.
    """
    try:
        float_num = float(number)
        integer_part = int(float_num)
        decimal_part = ""
        if float_num != integer_part:
            decimal_part = str(float_num).split('.')[-1]
            decimal_part = "." + decimal_part

        padded_integer = str(integer_part).zfill(4)
        return padded_integer + decimal_part
    except ValueError:
        return None