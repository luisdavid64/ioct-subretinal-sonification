import os
import subprocess
import time
import signal

def kill_process_processing(simulator_process):
    try:
        simulator_pid = simulator_process.pid
        simulator_pgid = os.getpgid(simulator_pid)
        print(f"🔄 Terminating Processing/Java simulator (PID: {simulator_pid}, PGID: {simulator_pgid})")
        
        # Step 1: Kill the entire process group immediately (Java apps need this)
        try:
            os.killpg(simulator_pgid, signal.SIGTERM)
            print("📡 Sent SIGTERM to process group")
            time.sleep(2)
            
            # Check if process group is still alive
            try:
                os.killpg(simulator_pgid, 0)  # Test if group exists
                print("⚠️  Process group still alive, escalating to SIGKILL")
                os.killpg(simulator_pgid, signal.SIGKILL)
                time.sleep(1)
            except (OSError, ProcessLookupError):
                print("✅ Process group terminated successfully")
                
        except (OSError, ProcessLookupError):
            print("✅ Process group already terminated")
            
        # Step 2: Specifically kill Java processes related to our simulator
        print("🔍 Killing any remaining Java/Processing processes...")
        java_kill_commands = [
            "pkill -f 'java.*processing_sonobox'",  # Java process running our app
            "pkill -f 'java.*Sonobox'",              # Alternative naming
            "pkill -f 'processing.*Sonobox'",         # Processing-specific
            "pkill -f 'MacOS/Sonobox'",              # macOS app bundle
        ]
        
        for cmd in java_kill_commands:
            try:
                result = os.system(f"{cmd} 2>/dev/null")
                if result == 0:
                    print(f"✅ Killed processes with: {cmd}")
            except:
                pass
                
        # Step 3: Final verification and cleanup
        time.sleep(1)
        try:
            simulator_process.poll()  # Update return code
            if simulator_process.returncode is None:
                print("⚠️  Main process still running, forcing kill")
                simulator_process.kill()
                simulator_process.wait(timeout=2)
            print("✅ Simulator termination completed")
        except subprocess.TimeoutExpired:
            print("❌ WARNING: Some processes may still be running")
            # Nuclear option for stubborn Java processes
            os.system("pkill -9 -f 'java.*processing' 2>/dev/null")
            os.system("pkill -9 -f 'Sonobox' 2>/dev/null")
            
    except Exception as e:
        print(f"❌ Error during termination: {e}")
        # Emergency cleanup
        os.system("pkill -9 -f 'java.*processing' 2>/dev/null")
        os.system("pkill -9 -f 'processing.*Sonobox' 2>/dev/null")
        os.system("pkill -9 -f 'Sonobox' 2>/dev/null")