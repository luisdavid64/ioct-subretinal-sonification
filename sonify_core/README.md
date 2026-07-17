# iOCT Sonification Runtime

Runtime code for `Physics-Based iOCT Sonification for Real-time Interaction Awareness in Subretinal Injection` (MICCAI 2026).

## Data format

Expected classes:

- `0`: Background
- `1`: Needle
- `2`: ILM
- `3`: RPE

## Running the code

### Physics-based iOCT sonification

```bash
python ioct_sonification.py /path/to/sequence --ranges RANGES_ID
```

This pipeline parses iOCT frames and drives the Processing-based sound model. Before running it, compile [../processing_sonobox](../processing_sonobox) and make sure [scripts/simulator_script.sh](./scripts/simulator_script.sh) points to the correct executable.

If segmentation masks are missing, the code falls back to the segmentation network included in this repository.

### ROS-based physics sonification

```bash
python ros_ioct_sonification.py
```

If you need a mock publisher for testing, see [scripts/ros_pub.sh](./scripts/ros_pub.sh).

### Distance-based sonification

```bash
python distance_ioct_sonification.py /path/to/sequence
```

This variant uses SuperCollider. Start the SuperCollider patch in [dist_sonification_main.scd](./dist_sonification_main.scd) before running the Python script.

## Notes

- ACE-related Python hooks live in [ace](./ace), but the corresponding ACE SuperCollider patch is not included in this repository.
- Example output videos are available in [../resources/samples](../resources/samples).
