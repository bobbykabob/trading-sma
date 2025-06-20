import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Real-Time Trading Dashboard...")
    
    # Install requirements
    if install_requirements():
        print("\n🌐 Starting web dashboard...")
        print("📱 The app will open in your browser automatically")
        print("🔄 Use Ctrl+C to stop the app")
        print("-" * 50)
        
        # Run the app
        run_streamlit_app()
    else:
        print("❌ Setup failed. Please check the error messages above.")
