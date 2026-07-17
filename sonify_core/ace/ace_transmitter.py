from pythonosc import udp_client
from pathlib import Path
import time
import datetime


class ACERecorder:
    """
    Control SuperCollider ACE recording via OSC.
    """

    def __init__(
        self,
        sc_ip="127.0.0.1",
        sc_port=57120,
        out_dir="recordings",
        prefix="ace",
    ):
        self.client = udp_client.SimpleUDPClient(sc_ip, sc_port)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.current_file = None

    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def start(self, filename=None):
        """
        Start recording.
        If filename is None, auto-generate one.
        """
        if filename is None:
            filename = f"{self.prefix}_{self._timestamp()}.wav"

        self.current_file = self.out_dir / filename

        self.client.send_message(
            "/ace/record/start",
            str(self.current_file)
        )

        return self.current_file

    def stop(self):
        """
        Stop recording.
        """
        self.client.send_message("/ace/record/stop", 1)

        return self.current_file

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
