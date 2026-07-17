"""
    This script generates a video of an audio file by
    creating a spectrogram visualization with a vertical line
    moving across the time axis in sync

"""

from PyQt5 import QtWidgets, QtCore, QtMultimedia
import sys
import pyqtgraph as pg
import librosa
import numpy as np

# CHANGE THE AUDIO PATH
audio_path = "/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/physics_based_f_i66_inference_27_02_2026_02_22_31/recording.wav"

# Load audio just for spectrogram
y, sr = librosa.load(audio_path, sr=None)
S = np.abs(librosa.stft(y))
S_db = librosa.amplitude_to_db(S, ref=np.max)
duration = len(y) / sr

app = QtWidgets.QApplication(sys.argv)

# ---- Spectrogram window ----
win = pg.GraphicsLayoutWidget(show=True)
plot = win.addPlot()
img = pg.ImageItem()
plot.addItem(img)

img.setImage(S_db.T)
img.setRect(0, 0, duration, sr/2)

# Limit frequency range to 8000 Hz
plot.setYRange(0, 8000)

line = pg.InfiniteLine(angle=90, pen='r')
plot.addItem(line)

import imageio
from PyQt5.QtGui import QImage

def export_video(output_path="spectrogram_output.mp4", fps=60):
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        audio_path=audio_path,
        audio_codec='aac'
    )

    total_frames = int(duration * fps)

    for frame in range(total_frames):
        t = frame / fps
        line.setPos(t)

        QtWidgets.QApplication.processEvents()

        # Grab widget
        qimg = win.grab().toImage()
        width = qimg.width()
        height = qimg.height()

        ptr = qimg.bits()
        ptr.setsize(height * width * 4)
        arr = np.array(ptr).reshape((height, width, 4))

        # writer.append_data(arr[:, :, :3])  # drop alpha
        rgb = arr[:, :, :3][:, :, ::-1]  # BGR -> RGB
        writer.append_data(rgb)

        print(f"Rendering frame {frame+1}/{total_frames}", end="\r")

    writer.close()
    print("\nDone.")

export_video()