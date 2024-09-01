import os
import subprocess
import urllib.request

def setup_ffmpeg():
    ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ffmpeg_dir = "/tmp/ffmpeg"
    
    if not os.path.exists(ffmpeg_dir):
        os.makedirs(ffmpeg_dir)
    
    ffmpeg_tar = os.path.join(ffmpeg_dir, "ffmpeg.tar.xz")
    
    # Download FFmpeg
    urllib.request.urlretrieve(ffmpeg_url, ffmpeg_tar)
    
    # Extract FFmpeg
    subprocess.run(["tar", "xf", ffmpeg_tar, "-C", ffmpeg_dir, "--strip-components=1"])
    
    # Add FFmpeg to PATH
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

# Call this function when your app starts
setup_ffmpeg()