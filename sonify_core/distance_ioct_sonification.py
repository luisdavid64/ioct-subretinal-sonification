import argparse
import json
import os
import shutil
import time
from time import sleep

import cv2
import numpy as np

from ace.postprocess_ace import postprocess_ace
from inference import preprocess_and_segment_bscan
from ioct_sonification_base import BaseIOCTSonification
from needle_tracker import NeedleTracker
from segmentation.segment_bscan import extrapolate_retina
from utils.sim_viz import (
    SonificationConfig,
    create_run_folder_and_save_data,
    move_recording_and_generate_spectrogram,
)
from utils.util import handle_video_controls

try:
    from dist_sonification_main import DistPulseEngine
    from distance_son import SCRecorder

    AUDIO_AVAILABLE = True
except ImportError:
    print("⚠️  dist_sonification_main or distance_son module not available.")
    AUDIO_AVAILABLE = False
    DistPulseEngine = None
    SCRecorder = None


sonification = None


def send_debug_message(message):
    print(f"Debug message: {message}")


def calculate_distance_to_next_layer(needle_tip_pos, ilm_line, rpe_line, anatomical_region):
    if needle_tip_pos[0] is None or needle_tip_pos[1] is None:
        return float("inf"), "none"

    tip_x = int(np.clip(needle_tip_pos[0], 0, len(ilm_line) - 1))
    tip_y = needle_tip_pos[1]

    ilm_y = ilm_line[tip_x] if not np.isnan(ilm_line[tip_x]) else None
    rpe_y = rpe_line[tip_x] if not np.isnan(rpe_line[tip_x]) else None

    if anatomical_region == "vitreous" and ilm_y is not None:
        return max(0, ilm_y - tip_y), "ILM"
    if anatomical_region == "retina" and rpe_y is not None:
        return max(0, rpe_y - tip_y), "RPE"
    if anatomical_region in {"ILM", "RPE"}:
        return 0.0, anatomical_region

    return float("inf"), "none"


def map_distance_to_pulse_frequency(distance, max_distance=100.0, min_freq=0.5, max_freq=10.0, min_distance=20.0):
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    frequency = max_freq - (normalized_distance * (max_freq - min_freq))
    return max(min_freq, frequency)


def send_pulse_frequency(frequency, base_tone_freq=440.0):
    global sonification

    if not AUDIO_AVAILABLE:
        return

    if sonification is None:
        try:
            sonification = DistPulseEngine()
            sonification.boot()
            print("🔊 Audio tone generation started")
        except Exception as exc:
            print(f"⚠️  Could not start audio: {exc}")
            return
    clamped_freq = max(100.0, min(base_tone_freq, 2000.0))
    clamped_pulse = max(0.1, min(frequency, 20.0))
    gain_db = -30.0  # Default gain level

    try:
        sonification.update_params(freq=clamped_freq, pulse_freq=clamped_pulse, gain_db=gain_db)
    except Exception as exc:
        print(f"⚠️  Error updating audio parameters: {exc}")


def stop_tone_generator():
    global sonification

    if sonification is None:
        return

    try:
        sonification.shutdown()
        sonification = None
        print("🔇 Audio tone generation stopped")
    except Exception as exc:
        print(f"⚠️  Error stopping audio: {exc}")


def get_needle_tip_pos_from_seg(needle_mask):
    if not np.any(needle_mask):
        return (None, None)

    ys, xs = np.where(needle_mask)
    if len(ys) == 0:
        return (None, None)

    max_y_idx = np.argmax(ys)
    tip_x, tip_y = xs[max_y_idx], ys[max_y_idx]
    return (float(tip_x), float(tip_y + 3))


class DistanceBasedIOCTSonifier(BaseIOCTSonification):
    def __init__(
        self,
        *,
        add_margin_to_roi=80,
        separate_ilm=True,
        ranges="INCREMENTAL",
        simulator_path="",
        target_resolution=(500, 500),
        max_needle_position_jump=100.0,
    ):
        super().__init__(
            num_nodes_x=5,
            num_nodes_y=40,
            extend_roi_to_needle_tip=False,
            add_margin_to_roi=add_margin_to_roi,
            sep_f_components=False,
            static_mapping_type="dClass",
            separate_ilm=separate_ilm,
            ranges=ranges,
            simulator_path=simulator_path,
            include_retina_in_ilm_rpe_drivers=False,
            thickness_statistic="median",
            use_confidence_weights=False,
            target_resolution=target_resolution,
            max_needle_position_jump=max_needle_position_jump,
        )

    def prepare_distance_initial_state(self, folder):
        initial_data = self.prepare_offline_initial_data(
            folder,
            self.add_margin_to_roi,
            fallback_loader=lambda frame_path, frame: preprocess_and_segment_bscan(frame_path),
        )
        if initial_data is None or initial_data.seg_img_0 is None:
            return None

        seg_img_0, ilm_line, rpe_line, _ = self.prepare_initial_segmentation(
            initial_data.seg_img_0,
            rpe_thickness_pixels=0,
        )
        needle_tip_pos = get_needle_tip_pos_from_seg(seg_img_0 == 1)
        return initial_data, seg_img_0, [ilm_line.copy(), rpe_line.copy()], needle_tip_pos

    def load_distance_segmentation(self, frame_file, seg_file):
        seg_img_current, _ = self.load_segmentation_from_source(
            seg_file,
            fallback_loader=lambda: preprocess_and_segment_bscan(frame_file),
        )
        if seg_img_current is None:
            return None

        seg_img_current = seg_img_current.copy()
        seg_img_current[seg_img_current == 4] = 0
        return extrapolate_retina(seg_img_current, cls_to_use=(4 if self.separate_ilm else 2))

    @staticmethod
    def detect_anatomical_region(needle_tip_pos, ilm_line, rpe_line, seg_img_current=None, tolerance=10):
        current_class = 0
        anatomical_region = "background"
        boundary_info = {"ilm_y": None, "rpe_y": None}

        if needle_tip_pos[0] is None or needle_tip_pos[1] is None:
            return current_class, anatomical_region, boundary_info

        tip_x = int(np.clip(needle_tip_pos[0], 0, len(ilm_line) - 1))
        tip_y = int(needle_tip_pos[1])

        ilm_y = ilm_line[tip_x] if not np.isnan(ilm_line[tip_x]) else None
        rpe_y = rpe_line[tip_x] if not np.isnan(rpe_line[tip_x]) else None
        boundary_info["ilm_y"] = ilm_y
        boundary_info["rpe_y"] = rpe_y

        if ilm_y is not None and rpe_y is not None:
            if tip_y < ilm_y - tolerance:
                return 0, "vitreous", boundary_info
            if abs(tip_y - ilm_y) <= tolerance:
                return 2, "ILM", boundary_info
            if ilm_y + tolerance < tip_y < rpe_y - tolerance:
                return 4, "retina", boundary_info
            return 3, "RPE", boundary_info

        if seg_img_current is not None and 0 <= tip_x < seg_img_current.shape[1] and 0 <= tip_y < seg_img_current.shape[0]:
            seg_class = int(seg_img_current[tip_y, tip_x])
            if seg_class != 1:
                anatomical_region = {
                    0: "vitreous",
                    2: "ILM",
                    3: "RPE",
                    4: "retina",
                }.get(seg_class, "background")
                current_class = seg_class

        return current_class, anatomical_region, boundary_info

    @staticmethod
    def map_region_to_tone(anatomical_region, current_class):
        if anatomical_region == "vitreous" or current_class == 0:
            return 200.0
        if anatomical_region == "ILM" or current_class == 2:
            return 800.0
        if anatomical_region == "retina" or current_class == 4:
            return 800.0
        if anatomical_region == "RPE" or current_class == 3:
            return 3400.0
        return 200.0

    @staticmethod
    def save_synced_video(run_folder_path, video_frames_buffer):
        if len(video_frames_buffer) <= 1:
            print("⚠️ No video frames recorded or insufficient frames for video creation")
            return None

        print(f"🎬 Creating synced video with {len(video_frames_buffer)} frames...")
        total_time = video_frames_buffer[-1][1] - video_frames_buffer[0][1]
        if total_time > 0:
            avg_framerate = len(video_frames_buffer) / total_time
            avg_framerate = max(5.0, min(avg_framerate, 60.0))
        else:
            avg_framerate = 30.0

        height, width = video_frames_buffer[0][0].shape[:2]
        video_filename = os.path.join(run_folder_path, "simulation_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, avg_framerate, (width, height))

        for frame, _ in video_frames_buffer:
            video_writer.write(frame)

        video_writer.release()

        timing_file = os.path.join(run_folder_path, "video_timing.json")
        timing_data = {
            "total_frames": len(video_frames_buffer),
            "total_duration_seconds": total_time,
            "average_framerate": avg_framerate,
            "frame_timestamps": [timestamp for _, timestamp in video_frames_buffer],
            "video_filename": "simulation_video.mp4",
        }
        with open(timing_file, "w") as file_handle:
            json.dump(timing_data, file_handle, indent=2)

        recording_source = "/tmp/rec_son.wav"
        recording_target = os.path.join(run_folder_path, "recording.wav")
        if os.path.exists(recording_source):
            shutil.move(recording_source, recording_target)

        print(f"📹 Synced video saved: {video_filename}")
        print(f"⏱️  Video duration: {total_time:.2f}s at {avg_framerate:.1f} FPS")
        print(f"📋 Timing data saved: {timing_file}")
        return video_filename


def parameterize_and_sonify_oct(
    folder,
    remap_with_segmentation=True,
    separate_ilm=True,
    refine_with_sam=None,
    ranges="INCREMENTAL",
    dynamic_segs=True,
    simulator_path="/Users/luisreyes/Sonify/SonifyOCT/processing_sonobox/macos-aarch64/processing_sonobox.app/Contents/MacOS/Sonobox",
    save_video=True,
    target_resolution=(500, 500),
):
    refine_with_sam = [] if refine_with_sam is None else [int(item) for item in refine_with_sam]

    print(f"Visualizing folder: {folder}")
    if "_seg" in os.path.basename(folder):
        return

    session = DistanceBasedIOCTSonifier(
        add_margin_to_roi=80,
        separate_ilm=separate_ilm,
        ranges=ranges,
        simulator_path=simulator_path,
        target_resolution=target_resolution,
    )
    prepared = session.prepare_distance_initial_state(folder)
    if prepared is None:
        print(f"No frames or initial segmentation available for {folder}. Skipping...")
        return

    initial_data, seg_img_0, spline_prior_lines, needle_tip_pos = prepared
    frame_files = initial_data.frame_files
    seg_files_path = initial_data.seg_files_path
    seg_files = initial_data.seg_files
    frame_0 = initial_data.frame_0

    if target_resolution is not None and (
        abs(initial_data.scale_x - 1.0) > 0.05 or abs(initial_data.scale_y - 1.0) > 0.05
    ):
        target_width, target_height = target_resolution
        print(
            f"🔄 Resizing from {initial_data.original_shape[1]}x{initial_data.original_shape[0]} "
            f"to {target_width}x{target_height}"
        )
        print(f"   Scale factors: x={initial_data.scale_x:.3f}, y={initial_data.scale_y:.3f}")

    print(f"Found {len(frame_files)} frames")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  B: Jump to beginning")
    print("  Left Arrow: Previous frame")
    print("  Right Arrow: Next frame")
    print("  G: Toggle Gaussian blur")
    print("  F: Toggle force display mode")
    print("  Escape: Exit")

    send_debug_message("Distance-based sonification initialized.")
    print("Starting Distance-Based Sonification")
    print("Beginning Distance-Based Sonification Loop")
    print("🔧 Initialized spline-based fitting:")
    print(f"   ILM line: {np.sum(~np.isnan(spline_prior_lines[0]))}/{len(spline_prior_lines[0])} valid points")
    print(f"   RPE line: {np.sum(~np.isnan(spline_prior_lines[1]))}/{len(spline_prior_lines[1])} valid points")

    tracker = NeedleTracker(
        init_frame=frame_0,
        tip_pos=needle_tip_pos,
        alpha=0.6,
        init_segmentation=seg_img_0 if "injection" not in folder.lower() else None,
    )
    print(f"Using {'segmentation' if dynamic_segs else 'template'}-based needle tracking")

    session.reset_tracking_state()
    session.start_spline_worker(debug=False)
    print("🚀 Started background spline fitting thread")

    recorder = SCRecorder() if AUDIO_AVAILABLE and SCRecorder is not None else None
    if save_video:
        print("📹 Dynamic framerate video recording enabled")

    sonification_start_time = time.time()
    class_changes_log = []
    video_frames_buffer = []
    video_start_time = None
    audio_started = False
    paused = False
    show_force = False
    debug = False
    index = 0

    try:
        while index < len(frame_files):
            frame_file, frame = session.load_frame_at_index(folder, frame_files, index)
            if frame is None:
                print(f"Could not load frame: {frame_file}")
                index += 1
                continue

            seg_img_current = None
            if remap_with_segmentation or dynamic_segs:
                seg_file = os.path.join(seg_files_path, seg_files[index]) if index < len(seg_files) else None
                seg_img_current = session.load_distance_segmentation(frame_file, seg_file)

            tracked_tip, _ = tracker.update(frame, segmentation_map=seg_img_current)
            needle_tip_pos = tracked_tip if tracked_tip is not None else (None, None)

            current_ilm_line = spline_prior_lines[0]
            current_rpe_line = spline_prior_lines[1]
            current_class, anatomical_region, boundary_info = session.detect_anatomical_region(
                needle_tip_pos,
                current_ilm_line,
                current_rpe_line,
                seg_img_current=seg_img_current,
            )

            if remap_with_segmentation and index % 5 == 0 and seg_img_current is not None:
                session.queue_spline_task(index, seg_img_current)

            for result in session.collect_recent_spline_results(index, max_age=10):
                seg_img_current = result["fitted_seg"]
                spline_prior_lines = [
                    result["updated_lines"]["ILM"].copy(),
                    result["updated_lines"]["RPE"].copy(),
                ]

            current_ilm_line = spline_prior_lines[0]
            current_rpe_line = spline_prior_lines[1]

            distance_to_next_layer, target_layer = calculate_distance_to_next_layer(
                needle_tip_pos,
                current_ilm_line,
                current_rpe_line,
                anatomical_region,
            )
            pulse_frequency = map_distance_to_pulse_frequency(
                distance_to_next_layer,
                max_distance=100.0,
                min_freq=1.0,
                max_freq=10.0,
            )
            base_tone_freq = session.map_region_to_tone(anatomical_region, current_class)
            send_pulse_frequency(pulse_frequency, base_tone_freq)

            if debug:
                info_y = 30
                cv2.putText(
                    frame,
                    f"Region: {anatomical_region}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Distance to {target_layer}: {distance_to_next_layer:.1f}px",
                    (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Pulse Frequency: {pulse_frequency:.1f}Hz",
                    (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Base Tone: {base_tone_freq:.0f}Hz",
                    (10, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                if boundary_info["ilm_y"] is not None and boundary_info["rpe_y"] is not None:
                    cv2.putText(
                        frame,
                        f"ILM: {boundary_info['ilm_y']:.0f}  RPE: {boundary_info['rpe_y']:.0f}",
                        (10, info_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow("Video Visualization Tool", frame)

            if save_video:
                start_callback = recorder.start if recorder is not None else (lambda: None)
                video_start_time, audio_started = session.buffer_video_frame(
                    video_frames_buffer,
                    frame,
                    video_start_time,
                    audio_started=audio_started,
                    start_audio_callback=start_callback,
                )
            elif not audio_started and recorder is not None:
                recorder.start()
                audio_started = True

            control_result = handle_video_controls(paused, index, len(frame_files), show_force)
            if control_result["exit"]:
                break

            paused = control_result["paused"]
            index = control_result["index"]
            show_force = control_result["show_force"]
    finally:
        cv2.destroyAllWindows()
        session.stop_spline_worker(timeout=2.0)

        if sonification is not None:
            try:
                sonification.stop()
            except Exception:
                pass

        if audio_started and recorder is not None:
            recorder.stop()

        stop_tone_generator()

    config = SonificationConfig(
        script_type=os.path.basename(folder),
        remap_with_segmentation=remap_with_segmentation,
        separate_ilm=separate_ilm,
        refine_with_sam=refine_with_sam,
        ranges=ranges,
        simulator_path=simulator_path,
        has_needle=True,
        is_synthetic=False,
        angle=0,
        needle_tip_pos=needle_tip_pos,
        masses=None,
        stiffnesses=None,
        damping=None,
        seg_img_0=seg_img_0,
    )
    run_folder_path = create_run_folder_and_save_data(
        folder="distance_based",
        sonification_start_time=sonification_start_time,
        class_changes_log=class_changes_log,
        sound_model_config=None,
        config=config,
        sample_name=os.path.basename(folder),
        model_type="distance_based",
    )

    if save_video:
        session.save_synced_video(run_folder_path, video_frames_buffer)

    sleep(2)

    recording_path = None
    try:
        recording_path = move_recording_and_generate_spectrogram(
            run_folder_path=run_folder_path,
            class_changes_log=class_changes_log,
            sonification_start_time=sonification_start_time,
        )
    except Exception as exc:
        print(f"Error during recording move/spectrogram generation: {exc}")

    if recording_path is None:
        run_recording_path = os.path.join(run_folder_path, "recording.wav")
        if os.path.exists(run_recording_path):
            recording_path = run_recording_path

    if recording_path is not None and os.path.exists(recording_path):
        postprocess_ace(os.path.abspath(recording_path))

    return run_folder_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance-based Sonification Tool")
    parser.add_argument("--folder_mode", default="single", help="Mode: 'single' for single folder, 'batch' for batch processing")
    parser.add_argument("root_folder", help="Path to the root folder containing video frame subfolders")
    parser.add_argument("--refine_with_sam", nargs="+", help="Masks to refine")
    parser.add_argument("--ranges", default="INCREMENTAL", help="Parameter ranges configuration")
    parser.add_argument(
        "--dynamic_segs",
        action="store_true",
        default=True,
        help="Use segmentation-based needle tracking instead of template matching",
    )
    parser.add_argument(
        "--simulator_path",
        default="/Users/luisreyes/Sonify/SonifyOCT/processing_sonobox/macos-aarch64/processing_sonobox.app/Contents/MacOS/Sonobox",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Save the visualization as an MP4 video file",
    )
    args = parser.parse_args()

    if args.folder_mode == "single":
        parameterize_and_sonify_oct(
            args.root_folder,
            refine_with_sam=args.refine_with_sam or [],
            ranges=args.ranges,
            dynamic_segs=args.dynamic_segs,
            simulator_path=args.simulator_path,
            save_video=args.save_video,
        )
    else:
        import glob

        folders = sorted(glob.glob(os.path.join(args.root_folder, "*", "*")))
        for folder_path in folders:
            if os.path.isdir(folder_path):
                parameterize_and_sonify_oct(
                    folder_path,
                    refine_with_sam=args.refine_with_sam or [],
                    ranges=args.ranges,
                    dynamic_segs=args.dynamic_segs,
                    simulator_path=args.simulator_path,
                    save_video=args.save_video,
                )
