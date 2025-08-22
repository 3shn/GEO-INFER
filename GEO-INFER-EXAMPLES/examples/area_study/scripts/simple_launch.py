#!/usr/bin/env python3
"""
Simple Streamlit Dashboard Launcher

Direct and simple launcher for the area study dashboard.
No complex subprocess management, just direct streamlit execution.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly.express
        print("✅ All dependencies found!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install streamlit pandas plotly")
        return False

def launch_dashboard():
    """Launch the dashboard using direct streamlit command."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_script = os.path.join(script_dir, 'dashboard_app.py')

    print("🏛️ GEO-INFER Area Study Dashboard")
    print("=" * 50)
    print("🚀 Launching dashboard...")
    print(f"📁 Script: {dashboard_script}")
    print("🌐 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)

    # Build streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        dashboard_script,
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--theme.base', 'light',
        '--browser.serverAddress', 'localhost',
        '--browser.serverPort', '8501'
    ]

    print(f"📋 Command: {' '.join(cmd)}")
    print()

    try:
        # Run streamlit directly
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    if not check_dependencies():
        return 1

    launch_dashboard()
    return 0

if __name__ == "__main__":
    sys.exit(main())
