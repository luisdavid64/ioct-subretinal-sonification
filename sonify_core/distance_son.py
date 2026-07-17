# from supercollider import Server, Synth
from pythonosc.udp_client import SimpleUDPClient    

class DistanceBasedSonification:
    def __init__(self, ip="127.0.0.1", port=57120):
        self.osc = SimpleUDPClient(ip, port)

    def update_params(self, freq, pulse_freq, gain=-20):
        self.osc.send_message("/son/update", [freq, pulse_freq, gain])

    def close(self):
        pass  # SC owns lifecycle

    def reset(self):
        self.osc.send_message("/son/update", [400, 0.5, -120])  # Mute sound


class SCRecorder:
    def __init__(self, ip="127.0.0.1", port=57120):
        self.client = SimpleUDPClient(ip, port)

    def start(self):
        self.client.send_message("/rec/start", [])

    def stop(self):
        self.client.send_message("/rec/stop", [])
