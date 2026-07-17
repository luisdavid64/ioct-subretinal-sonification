from pythonosc.udp_client import SimpleUDPClient


class OSC_Transmitt_Process:

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.client = SimpleUDPClient(ip, port)

    def oscTransmittProc_OSC_Param(self, line, oscMass=1):
        oscName_m = '/osc'
        oscID = f'm_{int(line[1])}_{int(line[0])}_{int(line[2])}'
        self.client.send_message(oscName_m, [oscID, float(oscMass)])

    def oscTransmittProc_SprD_Param(self, node1, node2, sprStf=0.01, sprDmp=0.0001):
        oscName_m = '/sprD'
        node1 = f'{int(node1[1])}_{int(node1[0])}_{int(node1[2])}'
        node2 = f'{int(node2[1])}_{int(node2[0])}_{int(node2[2])}'
        self.client.send_message(oscName_m, [node2, node1, float(sprStf), float(sprDmp)])

    def oscTransmittProc_RestLength_Param(self, node1, node2, restLength=10.0):
        oscName_m = '/restLength'
        node1 = f'{int(node1[1])}_{int(node1[0])}_{int(node1[2])}'
        node2 = f'{int(node2[1])}_{int(node2[0])}_{int(node2[2])}'
        self.client.send_message(oscName_m, [node2, node1, float(restLength)])

    def oscTransmittProc_remove_spring(self, node1, node2, sprStf=0.01, sprDmp=0.0001):
        oscName_m = '/remove/spring'
        node1 = f'{int(node1[1])}_{int(node1[0])}_{int(node1[2])}'
        node2 = f'{int(node2[1])}_{int(node2[0])}_{int(node2[2])}'
        self.client.send_message(oscName_m, [node2, node1])
    
    def oscTransmittProc_Toggle(self):
        oscName_t = '/toggle'
        self.client.send_message(oscName_t, None)
    def oscTransmittProc_Toggle_with_force(self, force_value):
        oscName_t = '/toggle'
        self.client.send_message(oscName_t, [force_value])
    def oscTransmittProc_Toggle_with_force_and_driver_idx(self, force_value, idx, listener_idx):
        oscName_t = '/toggle_and_move'
        self.client.send_message(oscName_t, [float(force_value), str(idx), str(listener_idx)])
    def ocsTransmittProc_MultiToggle(self, forces, ramp=False):
        oscName_t = '/multipleToggle'
        if ramp:
            oscName_t = '/multipleToggleRamp'
        self.client.send_message(oscName_t, forces)
    def ocsTransmittProc_MultiToggleSep(self, forces, ramp=False):
        oscName_t = '/multipleToggleSepComponents'
        if ramp:
            oscName_t = '/multipleToggleSepComponentsRamp'
        self.client.send_message(oscName_t, forces)
    def ocsTransmittProc_ToggleILM_RPE(self, f_ilm, f_rpe):
        oscName_t = "/toggle_ILM_RPE"
        self.client.send_message(oscName_t,[float(f_ilm), float(f_rpe)])

    def oscTransmittProc_Debug_message(self, message):
        oscName_d = '/debug'
        self.client.send_message(oscName_d, [str(message)])

    def oscTransmittProc_RecordStart(self):
        oscName_t = '/record/start'
        self.client.send_message(oscName_t, ["/Users/luisreyes/Sonify/SonifyOCT/utilities/inference/recording.wav"])

    def oscTransmittProc_RecordEnd(self):
        oscName_t = '/record/end'
        self.client.send_message(oscName_t, [])

    def oscTransmittProc_RescaleParams(self, stiff_mult, damp_mult):
        oscName_t = '/rescale/params'
        self.client.send_message(oscName_t, [float(stiff_mult), float(damp_mult)])

    def oscTransmittProc_FixNode(self, line):
        oscName_t = '/set/fixed'
        oscID = f'm_{int(line[1])}_{int(line[0])}_{int(line[2])}'
        self.client.send_message(oscName_t, [str(oscID)])

    def oscTransmittProc_MoveListener(self, line):
        oscName_t = '/move/listener'
        oscID = f'm_{int(line[1])}_{int(line[0])}_{int(line[2])}'
        self.client.send_message(oscName_t, [str(oscID)])
    
    def oscTransmittProc_SetCrackle(self, density, gain):
        oscName_t = '/set/crackle'
        self.client.send_message(oscName_t, [float(density), float(gain)])

    def oscTransmittProc_SetAMFM(self, def_value):
        oscName_t = '/set/amfm'
        self.client.send_message(oscName_t, [float(def_value)])
    def oscTransmittProc_ResetAMFM(self):
        oscName_t = '/reset/amfm'
        self.client.send_message(oscName_t, [])
    
    # Deformation noise system methods
    def oscTransmittProc_SetDeformation(self, active, intensity):
        """Enable/disable continuous deformation with intensity control"""
        oscName_t = '/set/deformation'
        self.client.send_message(oscName_t, [int(active), float(intensity)])
    
    def oscTransmittProc_TriggerDeformation(self, intensity, duration):
        """Trigger a deformation event with specified parameters"""
        oscName_t = '/trigger/deformation'
        self.client.send_message(oscName_t, [float(intensity), float(duration)])
    
    def oscTransmittProc_StopDeformation(self):
        """Stop all deformation noise"""
        oscName_t = '/stop/deformation'
        self.client.send_message(oscName_t, [])
    
    def oscTransmittProc_ConfigDeformation(self, gain, cutoff, resonance):
        """Configure deformation filter parameters"""
        oscName_t = '/config/deformation'
        self.client.send_message(oscName_t, [float(gain), float(cutoff), float(resonance)])
    
    # Jitter/instability control methods
    def oscTransmittProc_SetJitter(self, amplitude_jitter, cutoff_jitter, rate=5.0, enabled=True):
        """Set jitter parameters for audio instability effects"""
        oscName_t = '/set/jitter'
        self.client.send_message(oscName_t, [float(amplitude_jitter), float(cutoff_jitter), float(rate), int(enabled)])
    
    def oscTransmittProc_EnableJitter(self, enabled):
        """Enable/disable jitter effects"""
        oscName_t = '/enable/jitter'
        self.client.send_message(oscName_t, [int(enabled)])
    
    def oscTransmittProc_EnableLPF(self, enabled):
        """Enable/disable low-pass filter"""
        oscName_t = '/enable/lpf'
        self.client.send_message(oscName_t, [int(enabled)])
