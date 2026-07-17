from pythonosc.udp_client import SimpleUDPClient

c = SimpleUDPClient("127.0.0.1", 57121)
c.send_message("/ace/render", ["/Users/luisreyes/Sonify/SonifyOCT/utilities/runs/comparison/Injection04_massspring/recording.wav"])              # defaults