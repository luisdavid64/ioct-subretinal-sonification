void oscEvent(OscMessage msg) {
  String pattern = msg.addrPattern();
  switch(pattern) {

  case "/remove/spring": {
    String node1 = msg.get(0).stringValue();
    String node2 = msg.get(1).stringValue();
    String intID = "i_"+node1+"_"+node2;
    Interaction inter = model.getInteraction(intID);
    model.removeInteraction(inter);
    println("Removed interaction subset: " + intID);
    break;
  }

  case "/move/listener": {
    String listener_node = msg.get(0).stringValue();
    listeners.get(0).moveObserver(model.getMass(listener_node));
    println("Moved listener to node " + listeners.get(0).getMat().getName());
    break;
  }
  case "/rescale/params": {
    float stiff_mult = asFloat(msg, 0, 1);
    float damp_mult = asFloat(msg, 1, 1);
    println("Rescaling from baseline values by factor: " + stiff_mult + ", damping by factor: " + damp_mult);
    
    // Initialize baselines if not done already
    if (!baselinesInitialized) {
      initializeBaselines();
    }
    
    // Scale from baseline values, not current values
    for (Interaction inter : model.getInteractionList()) {
      String interID = inter.getName();
      if (baselineStiffness.containsKey(interID) && baselineDamping.containsKey(interID)) {
        float baselineK = baselineStiffness.get(interID);
        float baselineC = baselineDamping.get(interID);
        inter.setParam(param.STIFFNESS, baselineK * stiff_mult);
        inter.setParam(param.DAMPING, baselineC * damp_mult);
        System.out.println("Rescaled interaction " + interID + " to K: " + inter.getParam(param.STIFFNESS) + ", C: " + inter.getParam(param.DAMPING));
      }
    }
    break;
  }

  case "/set/fixed": {
    String fixedNodeName = msg.get(0).stringValue();
    Mass fixedMass = model.getMass(fixedNodeName);
    println(fixedNodeName);
    if (fixedMass != null) {
      // fixedMass.setType(massType.GROUND3D);
      model.changeToFixedPoint(fixedMass);
      println("Set mass " + fixedNodeName + " as FIXED.");
    } else {
      println("Mass " + fixedNodeName + " not found.");
    }
    break;
  }


  case "/osc":
    String oscID = msg.get(0).stringValue();
    float oscMass = msg.get(1).floatValue();
    Mass myM = model.getMass(oscID);
    myM.setParam(param.MASS, oscMass);
    println("OSC MSG RECEIVED:: mass " + oscID + " set to " + oscMass);
    break;
  case "/toggle_and_move":
  try {
    float force = msg.get(0).floatValue();
    String driver_node = msg.get(1).stringValue();
    String listener_node = msg.get(2).stringValue();
    drivers.get(0).moveDriver(model.getMass(driver_node));
    listeners.get(0).moveObserver(model.getMass(listener_node));
    println("### " + "applied force " + " with value " + force + " to node " + driver_node);
    drivers.get(0).applyFrc(force, force, force);
  } catch (Exception e) {
    println("Error processing /toggle_and_move message: " + e.getMessage());
  }
    break;
  case "/sprD":
  try {
    String node1 = msg.get(0).stringValue();
    String node2 = msg.get(1).stringValue();
    float intStifness = msg.get(2).floatValue();
    float intDamping = msg.get(3).floatValue();
    String intID = "i_"+node1+"_"+node2;
    Interaction inter = model.getInteraction(intID);
    inter.setParam(param.STIFFNESS, intStifness);
    inter.setParam(param.DAMPING, intDamping);
    println("SPRING MSG RECEIVED:: interaction " + intID + " set to K:" + intStifness + " C:" + intDamping);
  } catch (Exception e) {
    break;
  }
    break;

  case "/restLength":
  String intID = "";
  try {
    String node1 = msg.get(0).stringValue();
    String node2 = msg.get(1).stringValue();
    float restLength = msg.get(2).floatValue();
    intID = "i_"+node1+"_"+node2;
    Interaction inter = model.getInteraction(intID);
    inter.setParam(param.DISTANCE, restLength);
    println("REST LENGTH MSG RECEIVED:: interaction " + intID + " set to L0:" + restLength);
  } catch (Exception e) {
    println("Error setting rest length for interaction " + intID + ": " + e.getMessage());
    break;
  }
    break;

  case "/msgTrigger":
    triggerText = msg.get(0).stringValue();
    println("received trigger msg::" + triggerText);
    showText = true;  // Enable the text to be displayed
    textTimer = millis();  // Reset the timer
    break;
    
  case "/tissueToggle":
    ArrayList<String> stringIDs = new ArrayList<>();
    ArrayList<Float> forces = new ArrayList<>();
    int i = 0;
    int forceRampSteps = (int)(1. * 44100);
    stringIDs.add(msg.get(0).stringValue());  // firstID
    stringIDs.add(msg.get(2).stringValue());  // secondID
    stringIDs.add(msg.get(4).stringValue());  // thirdID

    forces.add(msg.get(1).floatValue());  // firstForce
    forces.add(msg.get(3).floatValue());  // secondForce
    forces.add(msg.get(5).floatValue());  // thirdForce
        
    for (int j = 0; j < Math.min(model.getDrivers().size(), stringIDs.size()); j++) {
        Driver3D driver = model.getDrivers().get(j);
        driver.moveDriver(model.getMass(stringIDs.get(j)));
        driver.applyFrc(forces.get(j), forces.get(j), forces.get(j));
        println("RECEIVED:: driver " + driver.getName() + " exciting node " + stringIDs.get(j) + " with force " + forces.get(j));
    }
    break;
  
  case "/multipleToggle":
      for (int k = 0; k < msg.arguments().length; k++) {
        float excF = msg.get(k).floatValue();
        if (excF == 0) continue; // skip zero-force commands
        Driver3D dr = drivers.get(k);
        // println("RECEIVED:: driver " + dr.getMat().getName() + " exciting with force " + excF);
        dr.applyFrc(0, excF, 0);
        //println("Driver Name: " + dr.getName());
        //println("excited with: " + excF);
        }
    break;
  case "/multipleToggleSepComponents":
    try {
      int nArgs = msg.arguments().length;
      // println("Received /multipleToggleSepComponents with " + nArgs + " arguments.");
      int groups = nArgs / 3; // each group is Fx,Fy,Fz
      int nDrivers = drivers.size();
      int count = Math.min(groups, nDrivers);

      if (groups * 3 != nArgs) {
        println("Warning: /multipleToggleSepComponents expects args in multiples of 3 (Fx,Fy,Fz). Ignoring trailing args.");
      }

      for (int k = 0; k < count; k++) {
        float fx = msg.get(k * 3 + 0).floatValue();
        float fy = msg.get(k * 3 + 1).floatValue();
        float fz = msg.get(k * 3 + 2).floatValue();
        if (fx + fy + fz == 0) continue; // skip zero-force commands
        Driver3D dr = drivers.get(k);
        // println("RECEIVED:: driver " + dr.getMat().getName() + " exciting with Fx=" + fx + " Fy=" + fy + " Fz=" + fz);
        dr.applyFrc(0, fy, 0);
      }
    } catch (Exception e) {
      println("Error processing /multipleToggleSepComponents: " + e.getMessage());
    }
    break;

  case "/toggle_ILM_RPE":
    float f_ILM = msg.get(0).floatValue();
    float f_RPE = msg.get(1).floatValue();
    int n_drivers = drivers_ILM.size();
    int n_pairs = n_drivers / 2;

    for (int j = 0; j < (1f * n_drivers)/ 16; j++) {
      Driver3D dr = drivers_ILM.get(j);
      float f_cur = (j > 1) ? f_ILM /2 : f_ILM;  // First half ILM, second half RPE
      dr.applyFrc(f_cur,f_cur,f_cur);
      println("RECEIVED:: ILM driver " + dr.getMat().getName() + " exciting with force " + f_ILM);
    }
    break;

  case "multipleToggleRamp":
      for (int k = 0; k < msg.arguments().length; k++) {
        float excF = msg.get(k).floatValue();
        if (excF == 0) continue; // skip zero-force commands
        Driver3D dr = drivers.get(k);
        println("RECEIVED:: driver " + dr.getMat().getName() + " exciting with force " + excF);
        dr.triggerForceRamp(excF, excF, excF, 44100/ 100);  // 1 second ramp
        //println("Driver Name: " + dr.getName());
        //println("excited with: " + excF);
        }
    break;
  case "multipleToggleSepComponentsRamp":
    try {
      int nArgs = msg.arguments().length;
      // println("Received /multipleToggleSepComponents with " + nArgs + " arguments.");
      int groups = nArgs / 3; // each group is Fx,Fy,Fz
      int nDrivers = drivers.size();
      int count = Math.min(groups, nDrivers);

      if (groups * 3 != nArgs) {
        println("Warning: /multipleToggleSepComponents expects args in multiples of 3 (Fx,Fy,Fz). Ignoring trailing args.");
      }

      for (int k = 0; k < count; k++) {
        float fx = msg.get(k * 3 + 0).floatValue();
        float fy = msg.get(k * 3 + 1).floatValue();
        float fz = msg.get(k * 3 + 2).floatValue();
        if (fx + fy + fz == 0) continue; // skip zero-force commands
        Driver3D dr = drivers.get(k);
        // println("RECEIVED:: driver " + dr.getMat().getName() + " exciting with Fx=" + fx + " Fy=" + fy + " Fz=" + fz);
        dr.triggerForceRamp(fx, fy, fz, 44100/100);
      }
    } catch (Exception e) {
      println("Error processing /multipleToggleSepComponents: " + e.getMessage());
    }
    break;
    
  case "/moveListener":
    String toMassName = msg.get(0).stringValue();
    String listenerName = msg.get(1).stringValue();
    println(" ### LIST MSG RECEIVED ####");
    for(Observer3D obs : model.getObservers()){
      if (obs.getName().equals(listenerName)) {  
            obs.moveObserver(model.getMass(toMassName));
            System.out.println("#####################  OBS moved to mass: " + toMassName);
            println("#######################################");
        }
    }
    break;
    
  case "/inertiaController":
    String setName = msg.get(0).stringValue();
    float setRadius = msg.get(1).floatValue();
    float setMass = msg.get(2).floatValue();  
    println("RECEIVED subset mod::" + setName + " with radius value:: " + setRadius);
    phys.setParamForMassSubset(setName, param.RADIUS, setRadius);
    phys.setParamForMassSubset(setName, param.MASS, setMass);
    break;

  case "/record/start": {
    println("Received /record/start command.");
    String savePath = "default_recording.wav"; // fallback
    if (msg.arguments().length > 0) {
      savePath = msg.get(0).stringValue();
    }
    startRecording(savePath);
    break;
  }

  case "/record/end": {
    
    println("Received /record/start command.");
    endRecording();
    break;
  }

case "/debug": {
    println("Debug Print Start");
    println("----Masses----");
    // Print model parameters: Mass, Stiffnesses and Damping
    model.getMassList().forEach(mass -> {
      double mVal = mass.getParam(param.MASS);
      println("Mass " + mass.getName() + " has M = " + mVal);
    });
    println("----Interactions----");
    model.getInteractionList().forEach(interaction -> {
      double kVal = interaction.getParam(param.STIFFNESS);
      double cVal = interaction.getParam(param.DAMPING);
      println("Interaction " + interaction.getName() + " has K = " + kVal + ", C = " + cVal);
    });
    println("Debug Print End");

    model.getDrivers().forEach(driver -> {
      println("Driver : " + driver.getMat().getName());
    });
    model.getObservers().forEach(listener -> {
      println("Listener : " + listener.getMat().getName());
    });

  }
    break;

  }

  // OSC ModelController API usage
  processOSCModelControllerEvents(pattern, msg);
    
}

void processOSCModelControllerEvents(String pattern, OscMessage msg) {

    // -------------------- CONTROLLER ROUTES (NEW) --------------------
  switch (pattern) {
    case "/dim/y": { // arg: +1 or -1
      int step = asInt(msg, 0, 0);
      if (step > 0) controller.incrementY();
      else if (step < 0) controller.decrementY();
      break;
    }

    case "/dim/x": { // arg: +1 or -1
      controller.adjustX(signOrZero(msg, 0));
      break;
    }

    case "/dim/z": { // arg: +1 or -1
      controller.adjustZ(signOrZero(msg, 0));
      break;
    }

    case "/mass/radius": { // arg: +1 or -1
      controller.adjustRadius(signOrZero(msg, 0));
      break;
    }

    case "/mass/m": { // arg: +1 or -1
      controller.adjustMs(asDoubleArray(msg));
      break;
    }

    case "/spring/k": { // arg: +1 or -1
      controller.adjustKs(asDoubleArray(msg));
      break;
    }

    case "/spring/c": { // arg: +1 or -1
      controller.adjustCs(asDoubleArray(msg));
      break;
    }

    case "/interaction/next": { // no args
      controller.cycleInteractionType();
      break;
    }
    case "/interaction/set": { // no args
      controller.setInteractionType(msg.get(0).stringValue());
      break;
    }

    case "/shiftInOut": { // arg: "X" or "Z"
      char axis = axisChar(msg, 0, 'X');
      controller.shiftDriversListeners(axis);
      break;
    }

    case "friction": {
      float delta = asFloat(msg, 0, 0);
      float mult = asFloat(msg, 1, 1);
      controller.adjustGlobalFriction(delta, mult);
    }

    case "resolution": {
      float mult = asFloat(msg, 1, 2);
      controller.adjustResolution(mult);
      break;
    }

    case "/set/crackle": {
      float density = asFloat(msg, 0, 0);
      float gain = asFloat(msg, 1, 1);
      audioStreamHandler.setCrackle(density, gain);
      break;
    }

    case "/set/jitter": {
      float amplitudeJitter = asFloat(msg, 0, 0);    // 0.0-1.0
      float cutoffJitter = asFloat(msg, 1, 0);       // 0.0-1.0  
      float rate = asFloat(msg, 2, 5);               // Hz
      boolean enabled = msg.arguments().length > 3 ? (msg.get(3).intValue() > 0) : true;
      audioStreamHandler.setJitter(amplitudeJitter, cutoffJitter, rate, enabled);
      println("Jitter set: amp=" + amplitudeJitter + " cutoff=" + cutoffJitter + " rate=" + rate + " enabled=" + enabled);
      break;
    }

    case "/enable/jitter": {
      boolean enabled = msg.get(0).intValue() > 0;
      audioStreamHandler.enableJitter(enabled);
      println("Jitter " + (enabled ? "enabled" : "disabled"));
      break;
    }

    case "/set/deformation": {
      boolean active = msg.get(0).intValue() > 0;
      float intensity = asFloat(msg, 1, 0.5);
      //audioStreamHandler.setDeformationNoise(active, intensity);
      println("Deformation noise: " + (active ? "ON" : "OFF") + " intensity: " + intensity);
      break;
    }
    
    case "/trigger/deformation": {
      float intensity = asFloat(msg, 0, 0.8);
      float duration = asFloat(msg, 1, 1.0);
      //audioStreamHandler.triggerDeformation(intensity, duration);
      println("Triggered deformation: intensity=" + intensity + " duration=" + duration);
      break;
    }
    
    case "/stop/deformation": {
      //audioStreamHandler.stopDeformation();
      println("Stopped deformation noise");
      break;
    }
    
    case "/config/deformation": {
      float gain = asFloat(msg, 0, 0.3);
      float cutoff = asFloat(msg, 1, 2000);
      float resonance = asFloat(msg, 2, 1.0);
      //audioStreamHandler.setDeformationParams(gain, cutoff, resonance);
      println("Deformation params: gain=" + gain + " cutoff=" + cutoff + " resonance=" + resonance);
      break;
    }

    case "/set/am": {
      float def = asFloat(msg, 0, 0);
      //audioStreamHandler.setAMFromDef(def);
      break;
    }
    case "/set/fm": {
      float def = asFloat(msg, 0, 0);
      //audioStreamHandler.setFMFromDef(def);
      break;
    }
    case "/set/amfm": {
      float def = asFloat(msg, 0, 0);
      // audioStreamHandler.setFMFromDef(def);
      // audioStreamHandler.setAMFromDef(def);
      break;
    }
    case "/reset/amfm": {
      // audioStreamHandler.setFM(0,0);
      // audioStreamHandler.setAM(0,0);
      break;
    }

    // -------------------- NON-CONTROLLER UTILITIES (OPTIONAL) --------------------

    case "/config/save": { // optional string arg: base path
      String name = (msg.typetag().length() > 0)
        ? msg.get(0).stringValue()
        : "/Users/luisreyes/Sonify/SonoBox/model_configs/config_api.json";
      config.writeProcessingJson(java.nio.file.Paths.get(name));
      System.out.println("Saved current configuration to Processing JSON format.");
      break;
    }

    case "/config/reset": { // optional string arg: base path
      resetConfig();
      break;
    }
  }
}

int signOrZero(OscMessage m, int i) {
  return Integer.signum(asInt(m, i, 0));
}

int asInt(OscMessage m, int i, int defVal) {
  try { return m.get(i).intValue(); } catch (Exception e) { return defVal; }
}

float asFloat(OscMessage m, int i, float defVal) {
  try { return m.get(i).floatValue(); } catch (Exception e) { return defVal; }
}

char axisChar(OscMessage m, int i, char defVal) {
  try {
    String s = m.get(i).stringValue();
    if (s == null || s.isEmpty()) return defVal;
    char c = Character.toUpperCase(s.charAt(0));
    return (c == 'X' || c == 'Z') ? c : defVal;
  } catch (Exception e) {
    return defVal;
  }
}

double[] asDoubleArray(OscMessage m) {
  String tags = m.typetag();          // e.g., "iii", "fff", "ifs", ...
  int n = tags.length();
  double[] out = new double[n];

  for (int i = 0; i < n; i++) {
    char t = tags.charAt(i);
    try {
      switch (t) {
        case 'i': out[i] = m.get(i).intValue();   break;  // 32-bit int
        case 'f': out[i] = m.get(i).floatValue(); break;  // 32-bit float
        case 'h': out[i] = (double) m.get(i).longValue();  break; // 64-bit int
        case 'd': out[i] = m.get(i).doubleValue();         break; // 64-bit float
        default:  out[i] = Double.NaN; // non-numeric (string, blob, etc.)
      }
    } catch (Exception e) {
      out[i] = Double.NaN;
    }
  }
  return out;
}

void startRecording(String path) {
  try {
    File f = new File(path);
    wav = new WavWriter(f, /*sr*/44100, /*channels*/2);
    wav.start();
    recTap = (block, nframes, sr, t) -> {
      try { wav.append(block, nframes); } catch (IOException e) { /* log */ }
    };
    audioStreamHandler.addTap(recTap);
    println("Recording started: " + f.getAbsolutePath());
  } catch (IOException e) {
    println("Failed to start recording: " + e.getMessage());
  }
}

void endRecording() {
  if (recTap != null) audioStreamHandler.removeTap(recTap);
  recTap = null;
  try {
    if (wav != null) { wav.stop(); println("Recording stopped and saved."); }
  } catch (IOException e) {
    println("Failed to finalize WAV: " + e.getMessage());
  }
  wav = null;
}

// Initialize baseline parameter storage
void initializeBaselines() {
  baselineStiffness = new HashMap<String, Float>();
  baselineDamping = new HashMap<String, Float>();
  
  // Store current parameter values as baselines
  for (Interaction inter : model.getInteractionList()) {
    String interID = inter.getName();
    float currentK = (float) inter.getParam(param.STIFFNESS);
    float currentC = (float) inter.getParam(param.DAMPING);
    
    baselineStiffness.put(interID, currentK);
    baselineDamping.put(interID, currentC);
  }
  
  baselinesInitialized = true;
  println("Baselines initialized for " + baselineStiffness.size() + " interactions");
}

// Optional: Function to reset baselines (call when you want to establish new baseline values)
void resetBaselines() {
  baselinesInitialized = false;
  initializeBaselines();
}
