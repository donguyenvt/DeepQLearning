import os
import subprocess
import sys
import time

def run_app():
    print("Starting the Deep Q-Learning Car application...")
    
    # Set environment variables for Kivy
    os.environ['KIVY_NO_ARGS'] = '1'
    os.environ['KIVY_WINDOW'] = 'egl_rpi'
    
    # Run the application
    try:
        subprocess.run([sys.executable, 'map.py'], check=True)
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Error running the application: {e}")

if __name__ == "__main__":
    run_app()