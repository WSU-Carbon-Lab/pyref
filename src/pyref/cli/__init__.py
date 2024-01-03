""" 
Command line interface for reflutils. The interface
allows for interacting with a database of experiments
and is intended to be used within a windows file
structure. 

@Author: Harlan Heilman

Intended to be used with the following structure:
Beamline -> Beamtime (MonthYY) - > Experiment Label (Refl, liquid, etc) -> Day (YYMMDD) -> Scan ID - > Data

The interface is intended to be used with the 
following unix style command system:

refl [command] [options] [arguments]

===============================================
Example usage:

1) Initialize the database:
```refl init```

command:
init - Initialize the database within the current folder, This will
create a .refl folder within the current directory and create a 
database file within that folder. The database file will contain a
.json file as a NO-SQL database. This also creates a .refl file in 
this directory that contains information about the location of the 
.refl folder and the database file. 

2) Add a new experiment/beamtime to the database:
```refl add -b 11.0.1.2 -y 2023 -m Nov -d 12 -e Refl```
or
```refl add -c -e Refl```

command:
add - Add a new experiment to the database, creating a .macro folder

options:
--current or -c: Add the current beamtime to the database
-b or --beamline: Beamline ID Default - 11.0.1.2
-y or --year: Year of beamtime Default - Current
-m or --month: Month of beamtime Default - Current
-d or --day: Day of beamtime Default - Current
-e or --experiment: Experiment type Default - Refl

3) Sort Beamtime Data:
```refl sort --current -e Refl -d 12```
or to sort all data for a given beamtime
```refl sort --current -e Refl --all```

command:
sort - Sort data from a given beamtime. Creates a new 
folder within the beamtime folder called .sorted and
creates a copy of the data within that folder. The copy
will have the file structure
Sample -> Energy -> Pol(Scan ID)

options:
--current or -c: Sort the current beamtime
-e or --experiment: Experiment type Default - Refl
-d or --day: Day of beamtime Default - Current
--all: Sort all data for a given beamtime

4) List Beamtime Data:
```refl list --current -e refl -d 12```
or to list all data for a given beamtime
```refl list --current -e refl --all```

command:
list - List data within a given beamtime folder
displaying it as a table with the following 
columns:
Scan Number | Sample Name | Elapsed Time | Pol | Energy |
if the --all option is used then it will display the 
following columns:
Day Number | Date | Number of Scans | Samples | Energies | Pol

options:
-c or --current: List the current beamtime
-e or --experiment: Experiment type Default - Refl
-d or --day: Day of beamtime Default - Current
--all: List all data for a given beamtime

5) Run The Main CLI:
```refl run```

command:
run - Run the main CLI for reflutils. This will 
allow the user to interact with the database. It will
ask you to choose a beamtime and then a day. It will then 
show all scans and thir simplified metadata. The user
can then asses if the scans are properly labeled, and sort them
The user can additional interface with the sorted data to 
perform analysis. This allows the user to quickly sort data, reduce it,
and comment on the results. The program will then create a .reduced
folder within the beamtime folder with a copy of all the reduced data. Added 
to the folder will be a .json file with comments from the user. These 
comments can then be accessed from the main CLI. 
"""

__app__name__ = "refl"
__version__ = "0.1.0"


(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_ERROR,
    BEAMLINE_ERROR,
    BEAMTIME_ERROR,
    EXPERIMENT_ERROR,
    DAY_ERROR,
    SCAN_ERROR,
    SAMPLE_ERROR,
) = range(10)

ERRORS = {
    SUCCESS: "Success",
    DIR_ERROR: "Directory Error",
    FILE_ERROR: "File Error",
    DB_ERROR: "Database Error",
    BEAMLINE_ERROR: "Beamline Error",
    BEAMTIME_ERROR: "Beamtime Error",
    EXPERIMENT_ERROR: "Experiment Error",
    DAY_ERROR: "Day Error",
    SCAN_ERROR: "Scan Error",
    SAMPLE_ERROR: "Sample Error",
}
