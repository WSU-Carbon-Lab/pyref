// Core rust functions for input output and error handling.


// --------------------------------------------------------------------------------------
// Core Rust structures for handling the a single beamtimes heaped data.
// --------------------------------------------------------------------------------------

struct ActiveBeamtime {
    // Struct to hold the path for the current active beamtime.
    // Responsible to watch for changes downstreme from the path.
    path: String,
    //
    experiments: Vec<ActiveExperiment>,
}

struct ActiveExperiment {
    // Struct to hold the path for the current active experimental data.
    // Responsible to watch for changes downstreme from the path.
    path: String,
    scans: Vec<Scan>,
}

struct Scan {
    // Struct to hold the path for the current active scan.
    // Responsible to watch for changes downstreme from the path.
    path: String,
    data: Vec<str>,
}

// --------------------------------------------------------------------------------------
// Implmentations and constructors for the above structures.
// --------------------------------------------------------------------------------------

impl ActiveBeamtime {
    // Constructor for ActiveBeamtime
    fn new(path: String) -> ActiveBeamtime {
        ActiveBeamtime {
            path,
            experiments: Vec
        }
    }
}
