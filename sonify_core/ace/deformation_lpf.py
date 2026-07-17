from pythonosc.udp_client import SimpleUDPClient
import numpy as np

c = SimpleUDPClient("127.0.0.1", 57120)

def map_deformation_and_confidence(
    deformation_norm,
    min_freq=800,
    max_freq=10000,
    max_jitter=800
):
    """
    deformation_norm: 0 (none) → 1 (max)
    confidence: 0 (low) → 1 (high)
    """

    deformation_norm = np.clip(deformation_norm, 0, 1)

    # Deformation → mean cutoff
    cutoff = max_freq - deformation_norm * (max_freq - min_freq)

    # Confidence → jitter amount (low confidence = more jitter)
    jitter = (1.0 - deformation_norm) * max_jitter

    return int(cutoff), int(jitter)


def send_lpf(cutoff, jitter):
    c.send_message("/lpf", [cutoff, jitter])
