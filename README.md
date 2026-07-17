# Physics-Based iOCT Sonification for Real-time Interaction Awareness in Subretinal Injection

Code for the paper `Physics-Based iOCT Sonification for Real-time Interaction Awareness in Subretinal Injection`, to appear at MICCAI 2026.

Luis D. Reyes Vargas*, Veronica Ruozzi*, Andrea K. M. Ross, Shervin Dehghani, Michael Sommersperger, Koorosh Faridpooya, Mohammad Ali Nasseri, Merle Fairhurst, Nassir Navab, and Sasan Matinfar
## Method Overview

![Method diagram](./resources/method_diagram.png)

## Overview

This repository contains our code for real-time intraoperative OCT sonification during subretinal injection. The main runtime lives in [sonify_core](./sonify_core/), the sound engine lives in [processing_sonobox](./processing_sonobox/), and helper visualization scripts live in [visualization](./visualization/).

## Resources

Example output samples are available in [resources/samples](./resources/samples/):

- `spectrogram_output.mp4`
- `spectrogram_output_pigeye.mp4`

## Setup

Create the environment and install dependencies:

```bash
conda create -n ioct_sonification
conda activate ioct_sonification
pip install --upgrade pip
pip install -r requirements.txt
```

ROS is optional, but needed for the ROS-based workflow.

## Running the code

The main entrypoints are documented in [sonify_core/README.md](./sonify_core/README.md).

In short:

1. Compile and configure the Processing-based sound engine in [processing_sonobox](./processing_sonobox/).
2. Update the simulator path in [sonify_core/scripts/simulator_script.sh](./sonify_core/scripts/simulator_script.sh) if needed.
3. Run one of the sonification pipelines from `sonify_core`.

Physics-based sonification:

```bash
cd sonify_core
python ioct_sonification.py /path/to/sequence --ranges RANGES_ID
```

ROS-based physics sonification:

```bash
cd sonify_core
python ros_ioct_sonification.py
```

Distance-based sonification:

```bash
cd sonify_core
python distance_ioct_sonification.py /path/to/sequence
```
