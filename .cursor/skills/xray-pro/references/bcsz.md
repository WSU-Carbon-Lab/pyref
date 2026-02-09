
# BCSz Core API

> Reference for: Xray Pro
> Load when: Interacting with beamline control systems via Python, using asyncio, zmq, or requiring hardware status and command definitions

---


## Overview

Reference for: BCS API client usage, communication protocols, and control/status interfaces for XRAY-PRO hardware

Documentation covers connection setup, event loop compatibility, ZMQ/asyncio integration, and motor/status enums.

---

```python

"""
A python client interface to the BCS API, using zmq and asyncio.

All API calls ultimately call bcs_request with specialized JSON. Calling bcs_request from client
application code is possible, but discouraged, and not supported, as the call signature may change in future versions.

Contact bcs@lbl.gov with questions / requests.
"""
import sys
import asyncio

if sys.platform[:3] == 'win':
    # zmq.asyncio does not support the default (proactor) event loop on windows.
    # so set the event loop to one zmq supports
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import zmq
import zmq.asyncio
import zmq.utils.z85

import json
import time

# helper classes
from enum import Flag  # for MotorStatus


class MotorStatus(Flag):
    HOME = 1
    FORWARD_LIMIT = 2
    REVERSE_LIMIT = 4
    MOTOR_DIRECTION = 8
    MOTOR_OFF = 16
    MOVE_COMPLETE = 32
    FOLLOWING_ERROR = 64
    NOT_IN_DEAD_BAND = 128
    FORWARD_SW_LIMIT = 256
    REVERSE_SW_LIMIT = 512
    MOTOR_DISABLED = 1024
    RAW_MOTOR_DIRECTION = 2048
    RAW_FORWARD_LIMIT = 4096
    RAW_REVERSE_LIMIT = 8192
    RAW_FORWARD_SW_LIMIT = 16384
    RAW_REVERSE_SW_LIMIT = 32768
    RAW_MOVE_COMPLETE = 65536
    MOVE_LT_THRESHOLD = 131072

    def is_set(self, flag):
        return bool(self._value_ & flag._value_)


# helper functions
def bytes_from_blob(blob):
    """Deserializes binary data blobs. This implementation uses zmq's Base85 encoding, but future ones may not."""
    blob_len = blob['length']
    blob_str = blob['blob']
    return zmq.utils.z85.decode(blob_str)[:blob_len]


_zmq_context = None


class BCSServer:
    """Represents a remote BCS endstation or beamline system, running the BCS zmq server.

    Each endpoint returns a dictionary of results. In addition to endpoint-specific keys, the following
    global keys are included.

    * **success** (*bool*) - API endpoint completion status. Note that this does *not* indicate endpoint errors.
    * **error_description** (*str*) - If the server failed to execute the endpoint, this should indicate why. There may be warnings or information here even if the endpoint completed successfully.
    * **log** (*bool*) - True if the server logged this request.

    """
    _zmq_socket = None

    @staticmethod
    async def _get_server_public_key(addr, port):
        clear_socket = _zmq_context.socket(zmq.REQ)
        clear_socket.connect(f'tcp://{addr}:{port}')
        await clear_socket.send('public'.encode())
        server_public = await clear_socket.recv()
        clear_socket.close()
        return server_public

    # def __init__(self, addr='127.0.0.1', port=5577):
    async def connect(self, addr='127.0.0.1', port=5577):

        """(formerly the Constructor) Supply the zmq address string, addr, to reach this endstation."""

        global _zmq_context

        # the first server object will create the global zmq context
        if not _zmq_context:
            _zmq_context = zmq.asyncio.Context()

        self._zmq_socket = _zmq_context.socket(zmq.REQ)

        (client_public_key, client_secret_key) = zmq.curve_keypair()

        # server_public_key = asyncio.get_running_loop().run_until_complete(self._get_server_public_key(addr, port))

        server_public_key = await self._get_server_public_key(addr, port)

        print(f'Server Public Key {server_public_key}')

        self._zmq_socket.setsockopt(zmq.CURVE_SERVERKEY, server_public_key)
        self._zmq_socket.setsockopt(zmq.CURVE_PUBLICKEY, client_public_key)
        self._zmq_socket.setsockopt(zmq.CURVE_SECRETKEY, client_secret_key)

        self._zmq_socket.connect(f'tcp://{addr}:{port + 1}')

    async def bcs_request(self, command_name, param_dict, debugging=False):
        """
        The method responsible for direct communication to the BCS server

        :param command_name: Name of the API endpoint
        :type command_name: str
        :param param_dict: Parameter dictionary
        :type param_dict: dict
        """
        if debugging:
            print(f"API command {command_name} BEGIN.")

        api_call_start = time.time()
        param_dict['command'] = command_name
        param_dict['_unused'] = '_unused'
        if 'self' in param_dict:
            del param_dict['self']
        await self._zmq_socket.send(json.dumps(param_dict).encode())
        response_dict = json.loads(await self._zmq_socket.recv())
        response_dict['API_delta_t'] = time.time() - api_call_start

        if debugging:
            print(f"API command {command_name} END {response_dict['API_delta_t']} s.")

        return response_dict

# end BCSz_header.py
    async def acquire_data(self, chans=[], time=0, counts=0) -> dict:
        """
        Acquires for ``time`` seconds, or ``counts`` counts. whichever is non-zero. **Waits for the acquision to complete** and returns data for channels specified in ``chans``.
        If both ``counts`` *and* ``time`` are non-zero, which parameter takes precedence is not defined.

        :param chans: AI channel names to acquire. An empty array will return data for all AI channels on the server.
        :type chans: list
        :param time: If non-zero, the amount of time to acquire.
        :type time: float
        :param counts: If non-zero, the number of counts to acquire.
        :type counts: int

        :return: Dictionary of results, with the following key(s).

            * **chans** (*list*) - The channel names that have data in the **data** array, in the same order.

            * **not_found** (*list*) - The requested channels in ``chans`` that were not found on the host system, if any.

            * **data** (*list*) - The acquired data, in the same order as the channel names in **chans**.


        """
        return await self.bcs_request('AcquireData', dict(locals()))

    async def at_preset(self, name="_none") -> dict:
        """
        Checks if the associated motor is at the preset position. The associated motor and preset position are defined on the server.

        :param name: Name of the preset (not the motor name).
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **at_preset** (*bool*) - True if the target motor is at the preset position.

            * **position** (*float*) - The preset position, in units of the target motor.


        """
        return await self.bcs_request('AtPreset', dict(locals()))

    async def at_trajectory(self, name="success") -> dict:
        """
        Checks if all requirement been satisfied to be "At trajectory" (usually just means that each motor is at its trajectory goal).

        :param name: Trajectory name (defined on the server).
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **at_trajectory** (*bool*) - Are the motors "At trajectory?"

            * **running** (*bool*) - Is the trajectory still running (are the motors still moving)?


        """
        return await self.bcs_request('AtTrajectory', dict(locals()))

    async def command_motor(self, commands=[], motors=[], goals=[]) -> dict:
        """
        Command one or more motors. A large number of commands are available.

        :param commands: Array of motor commands, one for each motor in ``motors``. Current valid commands are {None, Normal Move, Backlash Move, Velocity Move, Move to Home, Stop Motor, Set Position, Enable Motor, Disable Motor, Move to Index, Run Home Routine, Set Velocity, Set Acceleration, Set Deceleration, Enable and Move, Disable SW Limits, Enable SW Limits, Start Time Delay, Check Time Delay, Set Output Pulses, Backlash Jog, Normal Jog, Run Coord Program, Halt Coord Program, Gearing ON, Gearing OFF, Set Forward SW Limit, Set Reverse SW Limit, Revert Forward SW Limit, Revert Reverse SW Limit}.
        :type commands: list
        :param motors: Array of motor names
        :type motors: list
        :param goals: Goal (or associated value for the command) for each motor provided in ``motors``
        :type goals: list

        :return: Dictionary of results, with the following key(s).

            * **timed_out** (*list*) - List of motors that timed-out, if any.

            * **not_found** (*list*) - List of motors that were not found, if any.


        """
        return await self.bcs_request('CommandMotor', dict(locals()))

    async def current_scan_running(self) -> dict:
        """
        Returns the currently running 'integrated scan' run, or an empty string if none is running.



        :return: Dictionary of results, with the following key(s).

            * **running_scan** (*str*) - The name of the currently running scan, or an empty string if none.


        """
        return await self.bcs_request('CurrentScanRunning', dict(locals()))

    async def disable_breakpoints(self, name="") -> dict:
        """
        Disables the breakpoint output (output-on-position) of  motor controller for the named motor. Or, 'takes the motor out of flying mode'.

        :param name: Motor  name
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('DisableBreakpoints', dict(locals()))

    async def disable_motor(self, name="success") -> dict:
        """
        Disables the named motor.

        :param name: Motor name.
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('DisableMotor', dict(locals()))

    async def enable_motor(self, name="success") -> dict:
        """
        Enables the named motor.

        :param name: Motor name.
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('EnableMotor', dict(locals()))

    async def get_acquired(self, chans=[]) -> dict:
        """
        Retrieve the average value from the most recent *single-shot* acquisition (see StartAcquire).

        :param chans: AI channel name(s). An empty array returns data for all AI channels on the server.
        :type chans: list

        :return: Dictionary of results, with the following key(s).

            * **chans** (*list*) - AI channel names that have data in the **data** array.

            * **not_found** (*list*) - The requested channels in ``chans`` that were not found on the host system, if any.

            * **data** (*list*) - The acquired data, in the same order as the channel names in **chans**.


        """
        return await self.bcs_request('GetAcquired', dict(locals()))

    async def get_acquired_array(self, chans=[]) -> dict:
        """
        Retrieve the array acquired from the most recent *single-shot* acquisition (see StartAcquire).

        :param chans: AI channel name(s). An empty array returns data for all AI channels on the server.
        :type chans: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - The requested channels in ``chans`` that were not found on the server, if any.

            * **chans** (*list*) - Array of clusters/dictionaries. Each array element contains the channel (**chan**), **period**, **time**, and **data**. Data is an array of sampled values.


        """
        return await self.bcs_request('GetAcquiredArray', dict(locals()))

    async def get_acquire_status(self) -> dict:
        """
        Read the current acquisition state of the AI subsystem



        :return: Dictionary of results, with the following key(s).

            * **acquiring** (*bool*) - True if an acquisition is in progress

            * **started** (*bool*) - True if an acquisition has been started

            * **reading** (*bool*) - True if acquisition data is currently being retrieved from an acquisition device


        """
        return await self.bcs_request('GetAcquireStatus', dict(locals()))

    async def get_di(self, chans=[]) -> dict:
        """
        Get digital input (DI) channel value

        :param chans: DI channel name(s). An empty array returns data for all channels on the server.
        :type chans: list

        :return: Dictionary of results, with the following key(s).

            * **chans** (*list*) - DI channel names that have data in the **data** array.

            * **not_found** (*list*) - The requested channels in ``chans`` that were not found on the host system, if any.

            * **enabled** (*list*) - Boolean list, in the same order as **chans**, indicating if the channel is used (enabled).

            * **data** (*list*) - Boolean value of the DI's, in the same order as the channel names in **chans**.


        """
        return await self.bcs_request('GetDI', dict(locals()))

    async def get_flying_positions(self, name="") -> dict:
        """
        Retrieve the locations of motor **name** that will trigger acquisitions in a flying scan.

        :param name: Motor  name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **pulse_positions** (*list*) - Trigger (pulse) positions.


        """
        return await self.bcs_request('GetFlyingPositions', dict(locals()))

    async def get_folder_listing(self, path=".", pattern="*.txt", recurse=False) -> dict:
        """
        Get lists of all files and folders in a location descended from "C:\\\\Beamline Controls\\\\BCS Setup Data"

        :param path: Path to text file, relative to C:\\\\Beamline Controls\\\\BCS Setup Data. Use backslashes (\\\\) not forward slashes (/).
        :type path: str
        :param pattern: Pattern for files for which you want to search.
        :type pattern: str
        :param recurse: If True, list all descendent files and folders that match pattern.
        :type recurse: bool

        :return: Dictionary of results, with the following key(s).

            * **files** (*list*) - Array of paths, relative to C:\\\\Beamline Controls\\\\BCS Setup Data.

            * **folders** (*list*) - Array of paths, relative to C:\\\\Beamline Controls\\\\BCS Setup Data.


        """
        return await self.bcs_request('GetFolderListing', dict(locals()))

    async def get_freerun(self, chans=[]) -> dict:
        """
        Get freerun AI data for one or more channels

        :param chans: Array of channel names to get. An empty array retrieves all channels' data.
        :type chans: list

        :return: Dictionary of results, with the following key(s).

            * **chans** (*list*) - The retrieved channel names, in the same order as **data**.

            * **not_found** (*list*) - The channels requested in ``chans`` that were not found on the server.

            * **data** (*list*) - The retrieved data, corresponding to the channels in **chans**.


        """
        return await self.bcs_request('GetFreerun', dict(locals()))

    async def get_freerun_array(self, chans=[]) -> dict:
        """
        Retrieve most recent AI freerun data.

        :param chans: AI channel name(s). An empty array returns data for all AI channels on the server.
        :type chans: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - The requested channels in ``chans`` that were not found on the server, if any.

            * **chans** (*list*) - Array of clusters/dictionaries. Each array element contains the channel (**chan**), **data**, and **x_values**.


        """
        return await self.bcs_request('GetFreerunArray', dict(locals()))

    async def get_instrument_acquired1d(self, name="") -> dict:
        """
        Retrieve data (1D array) from the most recent acquisition.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **x** (*list*) - 1D array of abscissae

            * **y** (*list*) - 1D array of data


        """
        return await self.bcs_request('GetInstrumentAcquired1D', dict(locals()))

    async def get_instrument_acquired2d(self, name="") -> dict:
        """
        Retrieve data (2D array) from the most recent acquisition.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **data** (*list*) - 2D array of u32.


        """
        return await self.bcs_request('GetInstrumentAcquired2D', dict(locals()))

    async def get_instrument_acquired3d(self, name="") -> dict:
        """
        Retrieve data (3D array) from the most recent acquisition.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **data** (*list*) - 3D array of u32.


        """
        return await self.bcs_request('GetInstrumentAcquired3D', dict(locals()))

    async def get_instrument_acquisition_info(self, name="") -> dict:
        """
        Retrieve miscellanaeous info about the instrument and acquisition: file_name, sensor temperature (if applicable), live time, and dead time.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **file_name** (*str*) - Fully qualified path name to latest data file.

            * **temperature** (*float*) - The temperature of an internal element, such as a CCD, if the instrument supports it.

            * **live_time** (*list*) - Live time for the acquistion.

            * **dead_time** (*list*) - Dead time for the acquisition.


        """
        return await self.bcs_request('GetInstrumentAcquisitionInfo', dict(locals()))

    async def get_instrument_acquisition_status(self, name="") -> dict:
        """
        Retrieve all available status bits from the instrument subsystem for the named instrument.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **scan_params_set_up** (*bool*) - Are the scan parameters set up?

            * **acquiring** (*bool*) - Is the instrument acquiring data?

            * **single_shot** (*bool*) - Unclear at the moment. TBD

            * **aborted** (*bool*) - Is the acquisition aborted?

            * **data_available** (*bool*) - Is data available?

            * **ins_single_shot** (*bool*) - 'Instrument Single Shot': Also unclear. TBD.


        """
        return await self.bcs_request('GetInstrumentAcquisitionStatus', dict(locals()))

    async def get_instrument_count_rates(self, name="") -> dict:
        """
        Retreive instrument count rates

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **input_cr** (*list*) - Input count rate

            * **output_cr** (*list*) - Output count rate


        """
        return await self.bcs_request('GetInstrumentCountRates', dict(locals()))

    async def get_instrument_driver_status(self, name="") -> dict:
        """
        Returns the status of the BCS Instrument Driver for the named instrument.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **running** (*bool*) - True if the driver is running.


        """
        return await self.bcs_request('GetInstrumentDriverStatus', dict(locals()))

    async def get_motor(self, motors=[]) -> dict:
        """
        Get information and status for the motors in ``motors``

        :param motors: Array of motor names to retrieve.
        :type motors: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - The requested motors in ``motors`` that were not found on the host system, if any.

            * **data** (*list*) - Array of clusters/dictionaries. Each array element contains the motor name (**motor**), **position**, **position_raw**, **goal**, **goal_raw**, **status**, and **time**.

    **status** is a bit-field. TODO: helper function to test status


        """
        return await self.bcs_request('GetMotor', dict(locals()))

    async def get_motor_full(self, motors=[]) -> dict:
        """
        Returns the complete state of the requested motors.

        :param motors: Names of motors to query.
        :type motors: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - Names in **motors** that were not found, if any.

            * **data** (*list*) - dictionary of dictionaries all relateing to state. Too many parameters to list atm.


        """
        return await self.bcs_request('GetMotorFull', dict(locals()))

    async def get_panel_image(self, name="", quality=0) -> dict:
        """
        Send a current image of the LabVIEW panel, in jpg  format.

        :param name: Full path to panel
        :type name: str
        :param quality: JPEG quality 0-100
        :type quality: int

        :return: Dictionary of results, with the following key(s).

            * **image_blob** (*dict*) - Ideally opaque blob for holding binary data. Helper functions should be available to clients to extract blob contents. Contains **length** and **blob**.


        """
        return await self.bcs_request('GetPanelImage', dict(locals()))

    async def get_state_variable(self, name="") -> dict:
        """
        Get the value of the named BCS State Variable.

        :param name: Name of the state variable to retrieve
        :type name: str

        :return: Dictionary of results, with the following key(s).

            * **value** (*void*) - Value of the state variable

            * **found** (*bool*) - True if requested name was found on the server

            * **type** (*enum u16*) - 'String', 'Boolean', 'Integer', or 'Double'

            * **name** (*str*) - Same as ``name``


        """
        return await self.bcs_request('GetStateVariable', dict(locals()))

    async def get_subsystem_status(self) -> dict:
        """
        Returns the current status of all BCS subsystems on the server



        :return: Dictionary of results, with the following key(s).

            * **status** (*list*) - Array of dictionaryies of keys: **name** - Subsystem Name, and **status** - one of {Initializing, Running, Stopping, Stopped, Force Stop, Disabled, Bad Path, VI Broken Not in List}


        """
        return await self.bcs_request('GetSubsystemStatus', dict(locals()))

    async def get_text_file(self, path="You must have at least one input control in the Parameter cluster. Delete this (_unused) if you don't need it.") -> dict:
        """
        Get contents of a text file (usually acquired data) from any file in a location descended from "C:\\\\Beamline Controls\\\\BCS Setup Data".

        :param path: Path to text file, relative to C:\\\\Beamline Controls\\\\BCS Setup Data. Use backslashes (\\\\) not forward slashes (/).
        :type path: str

        :return: Dictionary of results, with the following key(s).

            * **text** (*str*) - Text of the file. Platform-dependent end-of-line characters have been converted to line feed characters


        """
        return await self.bcs_request('GetTextFile', dict(locals()))

    async def get_video_image(self, name="", quality=0, type="default") -> dict:
        """
        Return the most recent image from the named camera, in jpg format.

        :param name: Video (camera) name
        :type name: str
        :param quality: JPEG quality 0-100
        :type quality: int
        :param type: One of [default, roi, or threshold]
        :type type: enum u16

        :return: Dictionary of results, with the following key(s).

            * **image_blob** (*dict*) - Ideally opaque blob for holding binary data. Helper functions should be available to clients to extract blob contents. Contains **length** and **blob**.


        """
        return await self.bcs_request('GetVideoImage', dict(locals()))

    async def home_motor(self, motors=[]) -> dict:
        """
        Home the motors in the input array, ``motors``.

        :param motors: Names of the motors to home.
        :type motors: list

        :return: Dictionary of results, with the following key(s).

            * **timed_out** (*list*) - True if the home operation timed out.

            * **not_found** (*list*) - The motors in ``motors`` that are not found on the server, if any.


        """
        return await self.bcs_request('HomeMotor', dict(locals()))

    async def last_scan_run(self) -> dict:
        """
        Returns the last 'integrated scan' run, or an empty string if none have run.



        :return: Dictionary of results, with the following key(s).

            * **last_scan** (*str*) - The name of the last run scan, or an empty string if none.


        """
        return await self.bcs_request('LastScanRun', dict(locals()))

    async def list_ais(self) -> dict:
        """
        Return the list of all AI channels that are defined on the server.



        :return: Dictionary of results, with the following key(s).

            * **names** (*list*) - Llist of AI channels that are defined on the server. The list includes disabled and hidden channels.

            * **displayed** (*list*) - The list of AI channel that are displayed (not set to 'hidden' in BCS).

            * **disabled** (*list*) - Iindexes in **displayed** that are disabled.


        """
        return await self.bcs_request('ListAIs', dict(locals()))

    async def list_dios(self) -> dict:
        """
        Retrieves the complete list of digital input channel names defined on the server.



        :return: Dictionary of results, with the following key(s).

            * **names** (*list*) - Liist of all digital input channel names on the server. The list includes disabled and hidden channels.

            * **displayed** (*list*) - The list of channels that are displayed (not set to 'hidden' in BCS).

            * **disabled** (*list*) - Iindexes in **displayed** that are disabled.


        """
        return await self.bcs_request('ListDIOs', dict(locals()))

    async def list_instruments(self) -> dict:
        """
        Return the list of instruments that are defined on the server.



        :return: Dictionary of results, with the following key(s).

            * **names** (*list*) - The list of all instruments that are defined on the server,  including disabled and hidden instruments.

            * **displayed** (*list*) - The list of instruments that are displayed (not set to 'hidden' in BCS).

            * **disabled** (*list*) - Iindexes in **displayed** that are disabled.


        """
        return await self.bcs_request('ListInstruments', dict(locals()))

    async def list_motors(self) -> dict:
        """
        Return the list of motors that are defined on the server.



        :return: Dictionary of results, with the following key(s).

            * **names** (*list*) - The list of motors that are defined on the server. The list includes disabled and hidden motors.

            * **displayed** (*list*) - The list of motors that are displayed (not set to 'hidden' in BCS).

            * **disabled** (*list*) - Iindexes in **displayed** that are disabled.


        """
        return await self.bcs_request('ListMotors', dict(locals()))

    async def list_presets(self) -> dict:
        """
        Return array of motor preset positions. Each array entry is a dictionary describing one preset. The dictionary keys are **Preset Name**, **Motor Name**, **Preset Position**, **Tolerance (+/-)**.



        :return: Dictionary of results, with the following key(s).

            * **presets** (*list*) - Array of dictionaries, one per preset, each with keys: **Preset Name**, **Motor Name**, **Preset Position**, **Tolerance (+/-)**.


        """
        return await self.bcs_request('ListPresets', dict(locals()))

    async def list_state_variables(self) -> dict:
        """
        Get the complete list of state variable, and their types (Boolean, Integer, String, Double).



        :return: Dictionary of results, with the following key(s).

            * **names** (*list*) - Current list of state variable on the server.

            * **types** (*list*) - Variable types (Boolean, Integer, String, Double), in the same order as **names**.


        """
        return await self.bcs_request('ListStateVariables', dict(locals()))

    async def list_trajectories(self) -> dict:
        """
        List all trajectories.



        :return: Dictionary of results, with the following key(s).

            * **trajectories** (*list*) - Array of dictionaries, each element describing a trajectory.


        """
        return await self.bcs_request('ListTrajectories', dict(locals()))

    async def move_motor(self, motors=[], goals=[]) -> dict:
        """
        Command one or more motors to begin moves to supplied goals.

        :param motors: Array of motor names
        :type motors: list
        :param goals: Goal for each motor listed in ``motors``
        :type goals: list

        :return: Dictionary of results, with the following key(s).

            * **timed_out** (*list*) - List of motors that timed-out, if any.

            * **not_found** (*list*) - List of motors that were not found, if any.


        """
        return await self.bcs_request('MoveMotor', dict(locals()))

    async def move_to_preset(self, names=[]) -> dict:
        """
        Move to preset positions. Takes a list of preset names and executes them (sends their motors to their respective positions).

        :param names: Array of preset names to execute.
        :type names: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - List of names in ``names`` that were not found on the server, if any.


        """
        return await self.bcs_request('MoveToPreset', dict(locals()))

    async def move_to_trajectory(self, names=[]) -> dict:
        """
        Move to trajectory positions. Takes a list of trajectory names and executes them (sends their motors to their respective positions).

        :param names: Array of trajectory names to execute.
        :type names: list

        :return: Dictionary of results, with the following key(s).

            * **not_found** (*list*) - List of names in ``names`` that were not found on the server, if any.


        """
        return await self.bcs_request('MoveToTrajectory', dict(locals()))

    async def scan_status(self) -> dict:
        """
        Returns information about the  'integrated scan' system, namely the last scan (last_scan) run, the currently running scan, and the scanner status. An empty string indicates no scan.



        :return: Dictionary of results, with the following key(s).

            * **last_scan** (*str*) - The name of the last run scan, or an empty string if none.

            * **running_scan** (*str*) - The name of the currently running scan, or an empty string if none.

            * **scanner_status** (*str*) - The current status of the scanner. One of a large set of finite-state-machine states used by the scanner. The set also varies by scan. Empty string if the scanner has stopped running.

            * **log_directory** (*str*) - Location of data file, maybe.

            * **last_filename** (*str*) - Data file name

            * **user_path** (*str*) - Or maybe this is the location of the data file


        """
        return await self.bcs_request('ScanStatus', dict(locals()))

    async def set_breakpoints(self, name="", x0=0, dx=0, n=0, breakpoints=[], counts_or_units=False) -> dict:
        """
        Set Breakpoints for the named motor. Breakpoints are the positions at which the controller will generate acquisition triggers during a flying scan. Specify either {**x0**, **dx**, **n**} to generate a regular grid, or send an arbitrary list in **breakpoints**.

        :param name: Motor
        :type name: str
        :param x0: Starting position.
        :type x0: float
        :param dx: Interval spacing.
        :type dx: float
        :param n: Number of breakpoints (triggers) to generate.
        :type n: int
        :param breakpoints: An array of arbitrary locations to on which to trigger data acquisition.
        :type breakpoints: list
        :param counts_or_units: Are the locations in motor units or motor counts (False == units)?
        :type counts_or_units: bool

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('SetBreakpoints', dict(locals()))

    async def set_do(self, chan="success", value=False) -> dict:
        """
        Sets the digital output (DO) channel  ``chan`` to ``value``

        :param chan: Name of the channel to set.
        :type chan: str
        :param value: New value
        :type value: bool

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('SetDO', dict(locals()))

    async def set_motor_velocity(self, motors=[], velocities=[]) -> dict:
        """
        Set motor speeds. This endpoint duplicates the functionality of CommandMotor, and may be removed from future releases.

        :param motors: List of motors
        :type motors: list
        :param velocities: Speeds to set, in the same order as **motors**.
        :type velocities: list

        :return: Dictionary of results, with the following key(s).

            * **timed_out** (*list*) -

            * **not_found** (*list*) -


        """
        return await self.bcs_request('SetMotorVelocity', dict(locals()))

    async def set_state_variable(self, name="variable name", value=0) -> dict:
        """
        Set the value of the named BCS State Variable.

        :param name: Name of the state variable to set
        :type name: str
        :param value: New value of the state variable. Accepts string, numeric, and boolean values.
        :type value: int

        :return: Dictionary of results, with the following key(s).

            * **found** (*bool*) - True if requested name was found on the server


        """
        return await self.bcs_request('SetStateVariable', dict(locals()))

    async def start_acquire(self, time=0, counts=0) -> dict:
        """
        Start Acquisition for either ``time`` or ``counts``, whichever is non-zero. The acquisition is started, but the endpoint does not wait for the acquisition to complete.

        :param time: Acquisition time.
        :type time: float
        :param counts: Acquisition counts.
        :type counts: int

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StartAcquire', dict(locals()))

    async def start_flying_scan(self, name="", xi=0, xf=0, dx=0, speed=0) -> dict:
        """
        Start a flying scan with the named motor. The scan is started. The enpoint does not wait for the scan to finish.

        :param name: Motor  name
        :type name: str
        :param xi: Starting point
        :type xi: float
        :param xf: Stopping point
        :type xf: float
        :param dx: Interval between triggers (pulses)
        :type dx: float
        :param speed: Motor speed
        :type speed: float

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StartFlyingScan', dict(locals()))

    async def start_instrument_acquire(self, name="", run_type="Exposure", acq_time_s=0, acq_counts=0) -> dict:
        """
        Starts an instrument acquisition and waits for it to complete. Acquires for either **acq_time** seconds, or for **acq_counts** counts (triggers), whichever is non-zero. If both are non-zero, the result is undefined.

        :param name: Instrument name
        :type name: str
        :param run_type: One of {'Exposure', 'Total Counts'}. Determines whether to acquire for a set time, or a set number of counts/triggers.
        :type run_type: enum u32
        :param acq_time_s: Length of time for the acquisition.
        :type acq_time_s: float
        :param acq_counts: Number of counts or triggers to acquire for.
        :type acq_counts: int

        :return: Dictionary of results, with the following key(s).

            * **elapsed_s** (*float*) - Total duration of the acquisition in seconds.


        """
        return await self.bcs_request('StartInstrumentAcquire', dict(locals()))

    async def start_instrument_driver(self, name="") -> dict:
        """
        Starts the named instrument driver (does nothing if the driver is already running).

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StartInstrumentDriver', dict(locals()))

    async def stop_acquire(self, timeout_ms=0) -> dict:
        """
        Stop acquisition in progress. Waits for the acquisition to terminate within ``timeout_ms`` milleseconds. If the acquisition does not stop in that time, returns False in ``success`` and "timed out" in ``error_description``.

        :param timeout_ms: Time to wait for the acquisition to terminate, in milliseconds.
        :type timeout_ms: int

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StopAcquire', dict(locals()))

    async def stop_instrument_acquire(self, name="") -> dict:
        """
        Stop acquisition on the named instrument (does nothing if the instument is not acquiring). Waits up to half a second to verify that acquisition has terminated. The **success** field is False if the abort times out without terminating the acquisition.

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StopInstrumentAcquire', dict(locals()))

    async def stop_instrument_driver(self, name="") -> dict:
        """
        Stops the named instrument driver (does nothing if the driver is not running).

        :param name: Instrument name
        :type name: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StopInstrumentDriver', dict(locals()))

    async def stop_motor(self, motors=[]) -> dict:
        """
        Immediately issue stop command for the motors provided in ``motors``.

        :param motors: List of motors to stop. The stop command is immediately sent to the motor controller for the listed motors.
        :type motors: list

        :return: Dictionary of results, with the following key(s).

            * **timed_out** (*list*) - Motor subsystem error indicating that the command queue is unavailable. Seek help from the beamline scientist or beamline controls.

            * **not_found** (*list*) - Those motors requested in ``motors`` that were not found on the server, if any.


        """
        return await self.bcs_request('StopMotor', dict(locals()))

    async def stop_scan(self) -> dict:
        """
        Immediately issue stop command forthe currently running 'integrated scan'.



        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('StopScan', dict(locals()))

    async def test_connection(self, chan="success", value=False) -> dict:
        """


        :param chan:
        :type chan: str
        :param value:
        :type value: bool

        :return: Dictionary of results, with the following key(s).

            * **response_string** (*str*) -


        """
        return await self.bcs_request('TestConnection', dict(locals()))

    async def sc_alsu_mirror_vibration(self, time_sec=0, start_delay_sec=0, x_motor="", start_position=0, stop_position=0, duration_sec=0, delay_sec=0, final_position=0, left_sensor="", right_sensor="", seperation_mm=1, scale_factor_pm=0, description="", file_pattern="") -> dict:
        """
        Setup the ALSU Mirror Vibration Scan
        This scan moves one motor in a specific pattern and records the analog data. Additional calculations are performed on the Interferometer Data to provide more relevant calculations into the file.
        The setup that goes with this scan involves a mirror on a table. The mirror has cooling water flowed through it using a Mass Flow Controller. The "motor" in this scan is actually the flow of water through the mirror. The scan then is to flow water thorough the mirror and monitor the interferometer to look at the vibrations of the mirror.

        :param time_sec: Time that the motor moves for.
        :type time_sec: float
        :param start_delay_sec: Delay initial data acquisition. This will give a chance for the Flow controlled by the X motor to stabilize.
        :type start_delay_sec: float
        :param x_motor: Choose the motor that will move.
        :type x_motor: str
        :param start_position: Where the motor starts. This is generally flow, so the flow at which the experiement starts.
        :type start_position: float
        :param stop_position: Where the motor stops. This if generally flow, so the flow at which the experiment stops.
        :type stop_position: float
        :param duration_sec: Time that the data is acquired for.
        :type duration_sec: float
        :param delay_sec: Wait this long after acquisition starts before the motor move.
        :type delay_sec: float
        :param final_position: Position to set the X motor when finished with scan. This should be zero. That will stop the flow of liquid.
        :type final_position: float
        :param left_sensor: Left interferomter count channel.
        :type left_sensor: str
        :param right_sensor: Right Interferometer count channel.
        :type right_sensor: str
        :param seperation_mm: Distance seperating the two interferometers.
        :type seperation_mm: float
        :param scale_factor_pm: Scales the readings into pico meteres.
        :type scale_factor_pm: float
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_ALSU Mirror Vibration', dict(locals()))

    async def sc_auto_coll_single_axis_scan(self, direction="Forward", index=0, x_motor="Select Motor", start=-3, stop=0, increment=1, delay_after_move_s=0, count_time_s=0.5, number_of_scans=1, bidirect=True, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """
        This is VERY close to a Single Motor Scan. The only difference is how the files are saved. It has been changed so that a TCP command can specify the scans that are run with more control than just the regular Single motor scan. This was created for the Metrology lab.

        :param direction: Name the direction of the scan. This does NOT swap the start and end. It is only used to lable the file created.
        :type direction: enum u32
        :param index: Used to save this scan as part of a group of scans. A new folder will be created for every 0 index scan started. Subsequent scans with increasing indices will be added to the same folder.
        :type index: int
        :param x_motor: Select the name of the motor to move.
        :type x_motor: str
        :param start: Start the scan with X Motor here.
        :type start: float
        :param stop: Where to stop the scan.
        :type stop: float
        :param increment: How far to move between each sample.
        :type increment: float
        :param delay_after_move_s: How long to pause after each move.
        :type delay_after_move_s: float
        :param count_time_s: Time to acquire from data source to generate sample
        :type count_time_s: float
        :param number_of_scans: How many times should the motor make this motion during the scan.
        :type number_of_scans: int
        :param bidirect: If number of scans is more than 1, will move even scans in the opposite direction.
        :type bidirect: bool
        :param at_end_of_scan: Choose what the X Motor does at the scan end.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_AutoColl Single Axis Scan', dict(locals()))

    async def sc_barf_scan(self, instrument="", exposure_time_s=0.001, number_to_average=1, continuous_acquire=True, background_data="Q:\\Commissioning\\OffLineRuby\\background\\background2020-03-12_11-15-40.txt", fit_type="Lorentzian", ambient_peak_2=694.23, ambient_peak_1=692.8, description="", file_pattern="") -> dict:
        """
        The Barf scan acquires from a spectrometer and then fits the ruby peaks to estimate the pressure.

        :param instrument: Choose the instrumnet to acquire from.
        :type instrument: str
        :param exposure_time_s: Choose the exposure time for the instrument.
        :type exposure_time_s: float
        :param number_to_average: How many exposures to average.
        :type number_to_average: int
        :param continuous_acquire: Choose if the scan should continuously acquire. Starting a scan with this option will cause the scan to run forever. The user can then stop the scan with the stop button, or by unchecking this box.
        :type continuous_acquire: bool
        :param background_data: Background data to subtract from acquired data.
        :type background_data: str
        :param fit_type: Type of fit to do.
        :type fit_type: enum u32
        :param ambient_peak_2: Peak position of the 2nd peak when the pressure is zero.
        :type ambient_peak_2: float
        :param ambient_peak_1: Peak position of the 1st peak when pressure is zero.
        :type ambient_peak_1: float
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Barf Scan', dict(locals()))

    async def sc_from_file_scan(self, file_path="", delay_after_move_s=0, count_time_s=0.5, at_end="Return", move_motors_sequentially=False, dont_repeat_motor_moves=False, move_at_end_of_scan="", description="", file_pattern="") -> dict:
        """
        <p>This interface hides all the scan detail inside of a 'from file' scan file. Only the four remaining settings are used to affect every step listed in the file.</p>
        <p>The file contains motor names and positions. At each step, the motors will be set to those positions, then the data will be read. This allows the user to create scans of arbitrary complexity, but they are harder to setup because the file has to be created.</p>
        <p>Example:</p>
        <div class="example">
        <table>
        <tr>
        <td>Motor1</td>
        <td>Motor2</td>
        <td>Motor3</td>
        </tr>
        <tr>
        <td>10.0</td>
        <td>0.00</td>
        <td>0.00</td>
        </tr>
        <tr>
        <td>10.0</td>
        <td>10.0</td>
        <td>0.00</td>
        </tr>
        <tr>
        <td>0.00</td>
        <td>10.0</td>
        <td>10.0</td>
        </tr>
        </table>
        </div>
        <p>This is the template of the from file scan. For more complex behavior, use the trajectory scan.</p>

        :param file_path: Choose the file to run. File should have the motor names across the top and the positions on the rows below. Tab spaces values.
        :type file_path: str
        :param delay_after_move_s: After the motors move, the signal may need some time to settle. This can be added here to improve data quality.
        :type delay_after_move_s: float
        :param count_time_s: Specify how long to sample the input data for. All the data sampled during this time will be averaged together, depending on the data source.
        :type count_time_s: float
        :param at_end: If return to start is used, the motors will receive one last command to come back to the start after the scan has completed. Otherwise the motors will stay where it is left at the end of the scan.
        :type at_end: enum u32
        :param move_motors_sequentially: Move the motors one at a time.
        :type move_motors_sequentially: bool
        :param dont_repeat_motor_moves: Prevent identical positions from moving the motor twice. This could cause a motor to move when the desired action is for the motor to not move.
        :type dont_repeat_motor_moves: bool
        :param move_at_end_of_scan: Choose a Trajectory to run at the end of the scan.
        :type move_at_end_of_scan: str
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_From File Scan', dict(locals()))

    async def sc_image_from_file_scan(self, instrument="", save_images_as="", file_path="", delay_after_move_s=0, count_time_s=0.5, at_end="Return", move_motors_sequentially=False, dont_repeat_motor_moves=False, move_at_end_of_scan="", description="", file_pattern="") -> dict:
        """


        :param instrument:
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param file_path: Choose the file to run. File should have the motor names across the top and the positions on the rows below. Tab spaces values.
        :type file_path: str
        :param delay_after_move_s: After the motors move, the signal may need some time to settle. This can be added here to improve data quality.
        :type delay_after_move_s: float
        :param count_time_s: Specify how long to sample the input data for. All the data sampled during this time will be averaged together, depending on the data source.
        :type count_time_s: float
        :param at_end: If return to start is used, the motors will receive one last command to come back to the start after the scan has completed. Otherwise the motors will stay where it is left at the end of the scan.
        :type at_end: enum u32
        :param move_motors_sequentially: Move the motors one at a time.
        :type move_motors_sequentially: bool
        :param dont_repeat_motor_moves: Prevent identical positions from moving the motor twice. This could cause a motor to move when the desired action is for the motor to not move.
        :type dont_repeat_motor_moves: bool
        :param move_at_end_of_scan: Choose a Trajectory to run at the end of the scan.
        :type move_at_end_of_scan: str
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Image From File Scan', dict(locals()))

    async def sc_image_one_motor_scan(self, instrument="", save_images_as="", x_motor="Select Motor", start=-3, stop=0, increment=1, delay_after_move_s=0, count_time_s=0.5, number_of_scans=1, bidirect=True, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """


        :param instrument:
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param x_motor: Select the name of the motor to move.
        :type x_motor: str
        :param start: Start the scan with X Motor here.
        :type start: float
        :param stop: Where to stop the scan.
        :type stop: float
        :param increment: How far to move between each sample.
        :type increment: float
        :param delay_after_move_s: How long to pause after each move.
        :type delay_after_move_s: float
        :param count_time_s: Time to acquire from data source to generate sample
        :type count_time_s: float
        :param number_of_scans: How many times should the motor make this motion during the scan.
        :type number_of_scans: int
        :param bidirect: If number of scans is more than 1, will move even scans in the opposite direction.
        :type bidirect: bool
        :param at_end_of_scan: Choose what the X Motor does at the scan end.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Image One Motor Scan', dict(locals()))

    async def sc_image_time_scan(self, instrument="", save_images_as="", total_time_s=0, number_of_samples=0, time_between_points_s=0, count_time_s=0, frequency_hz=0, stop_condition="Samples Taken", description="", file_pattern="") -> dict:
        """


        :param instrument:
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param total_time_s: Time to run the scan in total.
        :type total_time_s: float
        :param number_of_samples: Samples to take in the Total Time.
        :type number_of_samples: int
        :param time_between_points_s: Will combine with Total time to set the number of samples. If Set is selected for Number of Samples, then the total time will change to acomplish the requested time between points.
        :type time_between_points_s: float
        :param count_time_s: Time that the data is averaged for to create a single data point.
        :type count_time_s: float
        :param frequency_hz: The user can specify the frequency in samples per second or number of samples. Both work with time and reconfigure the other.
        :type frequency_hz: float
        :param stop_condition: Should the scan end based on time or number of samples. Sometimes, the time between points cannot be accopmlished and the samples are taken at a longer interval. In cases where this is significant, the user may want to stop the scan based on time and not the number of samples taken.
        :type stop_condition: enum u16
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Image Time Scan', dict(locals()))

    async def sc_image_two_motor_scan(self, instrument="", save_images_as="", x_motor="", x_start=-3, x_stop=0, x_increment=1, delay_after_move_s=0, count_time_s=0.5, y_motor="", y_start=-3, y_stop=0, y_increment=1, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """


        :param instrument:
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param x_motor: Choose the motor to be moved as the X Motor.
        :type x_motor: str
        :param x_start: Place where X Motor will start scan.
        :type x_start: float
        :param x_stop: Furthest position of X Motor travel.
        :type x_stop: float
        :param x_increment: How far the X Motor moves with each X move.
        :type x_increment: float
        :param delay_after_move_s: How long to delay after each motor moves.
        :type delay_after_move_s: float
        :param count_time_s: Time that data is acquired and averaged to create sample.
        :type count_time_s: float
        :param y_motor: Choose the motor to be moved as the Y Motor.
        :type y_motor: str
        :param y_start: Place where Y Motor will start scan.
        :type y_start: float
        :param y_stop: Furthest position of Y Motor travel.
        :type y_stop: float
        :param y_increment: How far the Y Motor moves with each Y move.
        :type y_increment: float
        :param at_end_of_scan: Choose what the scan does upon completion. Stay at end will not command an end move. Return to start will.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Image Two Motor Scan', dict(locals()))

    async def sc_ltp_single_axis_scan(self, direction="Forward", index=0, instrument="", frames_to_average=0, delay_between_frames_s=0, x_motor="Select Motor", start=-3, stop=0, increment=1, delay_after_move_s=0, count_time_s=0.5, number_of_scans=1, bidirect=True, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """


        :param direction:
        :type direction: enum u32
        :param index:
        :type index: int
        :param instrument:
        :type instrument: str
        :param frames_to_average:
        :type frames_to_average: int
        :param delay_between_frames_s:
        :type delay_between_frames_s: float
        :param x_motor: Select the name of the motor to move.
        :type x_motor: str
        :param start: Start the scan with X Motor here.
        :type start: float
        :param stop: Where to stop the scan.
        :type stop: float
        :param increment: How far to move between each sample.
        :type increment: float
        :param delay_after_move_s: How long to pause after each move.
        :type delay_after_move_s: float
        :param count_time_s: Time to acquire from data source to generate sample
        :type count_time_s: float
        :param number_of_scans: How many times should the motor make this motion during the scan.
        :type number_of_scans: int
        :param bidirect: If number of scans is more than 1, will move even scans in the opposite direction.
        :type bidirect: bool
        :param at_end_of_scan: Choose what the X Motor does at the scan end.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_LTP Single Axis Scan', dict(locals()))

    async def sc_move_motor(self, motor_name="", motor_position=0, delay_after_move_sec=0, description="", file_pattern="") -> dict:
        """
        Setup this simple scan so that it will move a motor. This is used in the automation scripts to move things around.

        :param motor_name:
        :type motor_name: str
        :param motor_position: Position to move to.
        :type motor_position: float
        :param delay_after_move_sec: Scan will wait this long before moving completing.
        :type delay_after_move_sec: float
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Move Motor', dict(locals()))

    async def sc_move_trajectory(self, trajectory_name="", delay_after_move_sec=0, description="", file_pattern="") -> dict:
        """
        Setup this simple scan so that it will activate a trajectory. This not a trajectory scan. Instead the user needs to set up a motor trajectory which is a named configuration of motors moved together or in sequence.  This is used in the automation scripts to move things around.

        :param trajectory_name: Choose the name of the desired trajectory.
        :type trajectory_name: str
        :param delay_after_move_sec: The scan will wait this many seconds after moving.
        :type delay_after_move_sec: float
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Move Trajectory', dict(locals()))

    async def sc_powder_diffraction(self, exposure_time=0, number_of_samples=0, file_index=0, instrument="", save_images_as="", description="", file_pattern="") -> dict:
        """


        :param exposure_time:
        :type exposure_time: float
        :param number_of_samples:
        :type number_of_samples: int
        :param file_index:
        :type file_index: int
        :param instrument:
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Powder diffraction', dict(locals()))

    async def sc_set_dio(self, description="", file_pattern="") -> dict:
        """


        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Set DIO', dict(locals()))

    async def sc_single_crystal_scan(self, instrument="", save_images_as="", x_motor="", start=-3, stop=0, increment=1, count_time_s=0.5, description="", file_pattern="") -> dict:
        """
        Setup the single motor scan that also records an image from a ccd camera instrument, with each data point.
        This scan moves a single motor to multiple positions at regular intervals and takes data at each spot. The motor that is moved is recorded and is set as the default x axis when this data is graphed on an xy graph. If you look at this data on an Intensity plot or Image Graph, It will default to the "2D Image" data.

        :param instrument: Choose The Instrument to Capture Images From.
        :type instrument: str
        :param save_images_as:
        :type save_images_as: str
        :param x_motor: Selects the motor to scan.
        :type x_motor: str
        :param start: This is where the scan should start. The motor will move from here to the stop position during the scan.
        :type start: float
        :param stop: This is where the scan should stop. The motor will move from the start position to here during the scan.
        :type stop: float
        :param increment: This is specified in the the units of the motor. It determines how far to move the motor for each step.
        :type increment: float
        :param count_time_s: Specify how long to sample the input data for. All the data sampled during this time will be averaged together, depending on the data source.
        :type count_time_s: float
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Single Crystal Scan', dict(locals()))

    async def sc_single_motor_flying_scan(self, x_motor="", start=0, stop=10, increment=1, velocity_units=0.5, number_of_scans=1, bidirect=False, shift_ai=False, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """
        Setup a single motor flying scan.
        This scan makes one motor move and records data at specific positions during the move.
        To accomplish this, the scan sets up special conditions in the motor drive hardware to output pulses when the motor is at a certain position. When those pulses happen, the motor controller latches its current position into an array. The analog aquisition card also latches the analog values with every pulse.
        The scan polls the aquisition card and retreives all the data since the last poll. At the end of the scan, the motor positions are retreived and matched up with the analog data. This file is then saved.
        Because of how this operates, the data for the flying scan is not displayed until the end of the scan.

        :param x_motor: Choose the name of the motor to scan. This motor must support flying scans.
        :type x_motor: str
        :param start: Place where the first sample is taken. The scan will start elsewhere and accelerate to the proper velocity before the first sample is taken.
        :type start: float
        :param stop: Place where the last sample is taken. The scan will begin to decelerate after the last sample is taken.
        :type stop: float
        :param increment: The spacing between sample points.
        :type increment: float
        :param velocity_units: Specify how many times the motor moves along the patter for this scan. Cannot be 0.
        :type velocity_units: float
        :param number_of_scans: Will allow the scan to run back to front on even scans.
        :type number_of_scans: int
        :param bidirect: Specify the velocity in the underlying motor's units. This is not in counts.
        :type bidirect: bool
        :param shift_ai: Choose if the motor returns to the start at the end of the scan.
        :type shift_ai: bool
        :param at_end_of_scan: Choose behaviour at end of the scan.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Single Motor Flying Scan', dict(locals()))

    async def sc_single_motor_scan(self, x_motor="Select Motor", start=-3, stop=0, increment=1, delay_after_move_s=0, count_time_s=0.5, number_of_scans=1, bidirect=True, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """
        Setup the single motor scan.
        This scan moves a single motor to multiple positions at regular intervals and takes data at each spot. The motor that is moved is recorded and is set as the default x axis when this data is graphed on an xy graph.

        :param x_motor: Select the name of the motor to move.
        :type x_motor: str
        :param start: Start the scan with X Motor here.
        :type start: float
        :param stop: Where to stop the scan.
        :type stop: float
        :param increment: How far to move between each sample.
        :type increment: float
        :param delay_after_move_s: How long to pause after each move.
        :type delay_after_move_s: float
        :param count_time_s: Time to acquire from data source to generate sample
        :type count_time_s: float
        :param number_of_scans: How many times should the motor make this motion during the scan.
        :type number_of_scans: int
        :param bidirect: If number of scans is more than 1, will move even scans in the opposite direction.
        :type bidirect: bool
        :param at_end_of_scan: Choose what the X Motor does at the scan end.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Single Motor Scan', dict(locals()))

    async def sc_temperature_scan(self, instrument="", exposure_time_msec="", calibration_temperature="", xmin=0, xmax=0, lambda_max=0, lambda_min=0, nd_filter_motor="Select Motor", instrument_2="", exposure_time_msec_2="", calibration_temperature_2="", xmin_2=0, xmax_2=0, lambda_max_2=0, lambda_min_2=0, nd_filter_motor_2="Select Motor", instrument_3="", roi_size=0, temp_background_k=0, instrument_4="", roi_size_2=0, temp_background_k_2=0, acquire_images=False, description="", file_pattern="") -> dict:
        """
        The Temperature Scan acquires data from a spectrometer and then with background and calibration data calculates the temperature. It also optionally acquires an image from a 2D instrument and then displays a contour plot of the temperature profile along with recomputing the temperature based on the 2D image.

        :param instrument:
        :type instrument: str
        :param exposure_time_msec:
        :type exposure_time_msec: str
        :param calibration_temperature:
        :type calibration_temperature: str
        :param xmin:
        :type xmin: float
        :param xmax:
        :type xmax: float
        :param lambda_max:
        :type lambda_max: float
        :param lambda_min:
        :type lambda_min: float
        :param nd_filter_motor:
        :type nd_filter_motor: str
        :param instrument_2:
        :type instrument_2: str
        :param exposure_time_msec_2:
        :type exposure_time_msec_2: str
        :param calibration_temperature_2:
        :type calibration_temperature_2: str
        :param xmin_2:
        :type xmin_2: float
        :param xmax_2:
        :type xmax_2: float
        :param lambda_max_2:
        :type lambda_max_2: float
        :param lambda_min_2:
        :type lambda_min_2: float
        :param nd_filter_motor_2:
        :type nd_filter_motor_2: str
        :param instrument_3:
        :type instrument_3: str
        :param roi_size:
        :type roi_size: float
        :param temp_background_k:
        :type temp_background_k: float
        :param instrument_4:
        :type instrument_4: str
        :param roi_size_2:
        :type roi_size_2: float
        :param temp_background_k_2:
        :type temp_background_k_2: float
        :param acquire_images: Selects whether to also acquire the 2d images
        :type acquire_images: bool
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Temperature Scan', dict(locals()))

    async def sc_time_scan(self, total_time_s=0, number_of_samples=0, time_between_points_s=0, count_time_s=0, frequency_hz=0, stop_condition="Samples Taken", description="", file_pattern="") -> dict:
        """
        Setup the time scan.
        This scan takes periodic data aquisitions and stores them. The main parameters that need to be specified are: The number of samples, how often to sample, and how long to aquire data for. All other variables that can be configured relate back to these three.
        This scan supports the legacy header.

        :param total_time_s: Time to run the scan in total.
        :type total_time_s: float
        :param number_of_samples: Samples to take in the Total Time.
        :type number_of_samples: int
        :param time_between_points_s: Will combine with Total time to set the number of samples. If Set is selected for Number of Samples, then the total time will change to acomplish the requested time between points.
        :type time_between_points_s: float
        :param count_time_s: Time that the data is averaged for to create a single data point.
        :type count_time_s: float
        :param frequency_hz: The user can specify the frequency in samples per second or number of samples. Both work with time and reconfigure the other.
        :type frequency_hz: float
        :param stop_condition: Should the scan end based on time or number of samples. Sometimes, the time between points cannot be accopmlished and the samples are taken at a longer interval. In cases where this is significant, the user may want to stop the scan based on time and not the number of samples taken.
        :type stop_condition: enum u16
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Time Scan', dict(locals()))

    async def sc_trajectory_scan(self, file_path="", delay_after_move_s=0, count_time_s=0.5, at_end="Return", move_motors_sequentially=False, dont_repeat_motor_moves=False, shift_flying_data=False, move_at_end_of_scan="", description="", file_pattern="") -> dict:
        """
        <p>This simple interface hides all the scan detail inside of a tracjectory scan file. Only the three remaining settings are used to affect every step listed in the file.</p>
        <p>The file contains motor names and positions. At each step, the motors will be set to those positions, then the data will be read. This allows the user to create scans of arbitrary complexity, but they are harder to setup because the file has to be created.</p>
        <p>Example:</p>
        <div class="example">
        <table>
        <tr>
        <td>Motor1</td>
        <td>Motor2</td>
        <td>Motor3</td>
        </tr>
        <tr>
        <td>10.0</td>
        <td>0.00</td>
        <td>0.00</td>
        </tr>
        <tr>
        <td>10.0</td>
        <td>10.0</td>
        <td>0.00</td>
        </tr>
        <tr>
        <td>0.00</td>
        <td>10.0</td>
        <td>10.0</td>
        </tr>
        </table>
        </div>
        <p>There are many more options for the files.</p>

        :param file_path: Specify the path to the trajectory scan file.
        :type file_path: str
        :param delay_after_move_s: After the motors move, the signal may need some time to settle. This can be added here to improve data quality.
        :type delay_after_move_s: float
        :param count_time_s: Specify how long to sample the input data for. All the data sampled during this time will be averaged together, depending on the data source.
        :type count_time_s: float
        :param at_end: If return to start is used, the motors will receive one last command to come back to the start after the scan has completed. Otherwise the motors will stay where it is left at the end of the scan.
        :type at_end: enum u32
        :param move_motors_sequentially: Move the motors one at a time.
        :type move_motors_sequentially: bool
        :param dont_repeat_motor_moves: Prevent identical positions from moving the motor twice. This could cause a motor to move when the desired action is for the motor to not move.
        :type dont_repeat_motor_moves: bool
        :param shift_flying_data: Shift the flying dat ahead one time interval.
        :type shift_flying_data: bool
        :param move_at_end_of_scan: A trajectory to be called at the end of the scan. Trajectories can be setup in the motor system. The BCS team can help. If the trajectory is left blank, nothing will be done.
        :type move_at_end_of_scan: str
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Trajectory Scan', dict(locals()))

    async def sc_two_motor_scan(self, x_motor="", x_start=-3, x_stop=0, x_increment=1, delay_after_move_s=0, count_time_s=0.5, y_motor="", y_start=-3, y_stop=0, y_increment=1, at_end_of_scan="Return", description="", file_pattern="") -> dict:
        """
        This is the display for the setup when the scan is in the "View Data" mode. It has the same fields as the "Run Scan" mode, but everything is for display only.

        :param x_motor: Choose the motor to be moved as the X Motor.
        :type x_motor: str
        :param x_start: Place where X Motor will start scan.
        :type x_start: float
        :param x_stop: Furthest position of X Motor travel.
        :type x_stop: float
        :param x_increment: How far the X Motor moves with each X move.
        :type x_increment: float
        :param delay_after_move_s: How long to delay after each motor moves.
        :type delay_after_move_s: float
        :param count_time_s: Time that data is acquired and averaged to create sample.
        :type count_time_s: float
        :param y_motor: Choose the motor to be moved as the Y Motor.
        :type y_motor: str
        :param y_start: Place where Y Motor will start scan.
        :type y_start: float
        :param y_stop: Furthest position of Y Motor travel.
        :type y_stop: float
        :param y_increment: How far the Y Motor moves with each Y move.
        :type y_increment: float
        :param at_end_of_scan: Choose what the scan does upon completion. Stay at end will not command an end move. Return to start will.
        :type at_end_of_scan: enum u32
        :param description:
        :type description: str
        :param file_pattern:
        :type file_pattern: str

        :return: Dictionary of results (no endpoint-specific keys).



        """
        return await self.bcs_request('sc_Two Motor Scan', dict(locals()))

```
