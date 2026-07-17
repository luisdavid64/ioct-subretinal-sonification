import numpy as np

BSCAN_NAME = 'sim1'

ASCAN_PER_SECOND = 50
OSC_SERVER_ADDRESS = "192.168.50.202"
SERVER_PORT = 1000
PLOT_RANGE = 200

mass_per_bucket = 2

# Mass, Stiffness, Damping
# first = np.array([[5., 10.], [2., 3.], [0.0001, 0.5]])
# second = np.array([[8, 10.], [3., 5.], [0.0001, 0.5]])
first  = np.array([[5., 10.], [3,1], [0.0001, 0.003]])
second = np.array([[5., 10.], [5,3], [0.0001, 0.003]])

RANGE_PARAMS  = np.array([[1., 3.], [0.5,2]])

# Mass Up, Stiffness Up, Damping Down
INCREMENTAL = [
    [[25., 40.], [0.005, 0.5], [0.1, 0.0001]],
    [[20., 25.], [0.5, 1], [0.05, 0.001]],
    [[10., 20.], [1.0, 2], [0.01, 0.01]],
    [[10., 15.], [3.0, 2], [0.005, 0.005]],
    [[15., 25.], [3, 4], [0.05, 0.001]],
]

SETUP_OPTIMAL = [
    # class 0 – background: dull and quiet
    [[25., 25.], [0.5, 0.1], [0.003, 0.1]], # bg

    # class 1 – ignored
    [[50., 50.], [1.0, 1.0], [0.01, 0.01]],

    # class 2 – ILM: piercing detection, sharp but not overpowering RPE
    [[30., 30.], [4.0, 4.0], [0.0002, 0.0002]],

    # class 3 – retina: INCREASED stiffness to not filter high frequencies from RPE
    [[70., 40.], [4.0, 1], [0.0002, 0.002]],  # Changed from 0.5 to 2.0

    # class 4 – RPE: **danger**, must be highest pitch and well audible
    [[25.,25.], [5.0, 5.0], [0.0001, 0.0001]],
]

SETUP_SEG_SPLINES = [
    # class 0 – background (quiet, heavy, overdamped)
    [[300., 300.],   [0.5, 0.5],   [0.5, 0.5]],

    # class 1 – ignored (almost silent)
    [[120., 120.], [0.1, 0.1],   [0.04, 0.04]],

    # class 2 – ILM (sharp, brittle, fast decay)
    [[40., 40.],   [4.0, 4.0],   [0.015, 0.015]],

    # class 3 – retina (resonant, sustained, “safe zone”)
    [[100., 100.],   [1.0, 1.0],   [0.1, 0.1]],

    # class 4 – RPE / danger (high tension, unstable decay)
    [[50., 50.],   [5.0, 5.0],   [0.006, 0.006]],
]
# SETUP_SEG_SPLINES = [
#     # class 0 – background (quiet, heavy, overdamped)
#     [[96., 96.],   [0.25, 0.25],   [0.5, 0.5]],

#     # class 1 – ignored (almost silent)
#     [[90., 90.], [0.1, 0.1],   [0.04, 0.04]],

#     # class 2 – ILM (sharp, brittle, fast decay)
#     [[96., 96.],   [2.0, 2.0],   [0.0015, 0.0015]],

#     # class 3 – retina (resonant, sustained, “safe zone”)
#     [[40., 40.],   [0.5, 0.5],   [0.05, 0.05]],

#     # class 4 – RPE / danger (high tension, unstable decay)
#     [[96., 96.],   [4, 4],   [0.0001, 0.0001]],
# ]

SETUP_SEG_SPLINES = [
    # class 0 – background (quiet, heavy, overdamped)
    [[300., 300.],   [0.5, 0.5],   [0.5, 0.5]],

    # class 1 – ignored (almost silent)
    [[120., 120.], [0.1, 0.1],   [0.04, 0.04]],

    # class 2 – ILM (sharp, brittle, fast decay)
    [[80., 80.],   [2.5, 2.5],   [0.02, 0.08]],

    # class 3 – retina (resonant, sustained, “safe zone”)
    [[100., 100.],   [1.0, 1.0],   [0.01, 0.01]],

    # class 4 – RPE / danger (high tension, unstable decay)
    # Implemented: RPE changes as you go deeper
    [[60., 80.],   [10.0, 8.0],   [0.005, 0.02]],
]


RANGE_PARAMS_ALL = {
    "INCREMENTAL": INCREMENTAL,
    "SETUP_OPTIMAL": SETUP_OPTIMAL,
    "SETUP_SPLINES": SETUP_SEG_SPLINES
}

mapping_names = {
    0: "background",
    1: "Needle",
    2: "ILM",
    3: "RPE",
    4: "Retina"
}

names_to_mapping = {
    "Background": 0,
    "Needle": 1,
    "ILM": 2,
    "RPE": 3,
    "Retina": 4
}

mapping_syntheseyes = {
    0:0,
    1:2,
    2:1,
    3:3,
    4:0,
    5:0
}

mapping_classes = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4
}