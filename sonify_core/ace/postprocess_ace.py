from pythonosc.udp_client import SimpleUDPClient

def postprocess_ace(input_wav_path):
    c = SimpleUDPClient("127.0.0.1", 57121)
    c.send_message("/ace/render", [input_wav_path])