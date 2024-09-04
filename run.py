from app import app
from setup_ffmpeg import setup_ffmpeg

if __name__ == "__main__":
    setup_ffmpeg()
    app.run(debug=True)