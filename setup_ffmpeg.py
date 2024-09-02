import os
import subprocess
import urllib.request

def setup_ffmpeg():
    ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    ffmpeg_dir = "/tmp/ffmpeg"
    
    if not os.path.exists(ffmpeg_dir):
        os.makedirs(ffmpeg_dir)
    
    ffmpeg_tar = os.path.join(ffmpeg_dir, "ffmpeg.tar.xz")

    urllib.request.urlretrieve(ffmpeg_url, ffmpeg_tar)

    subprocess.run(["tar", "xf", ffmpeg_tar, "-C", ffmpeg_dir, "--strip-components=1"])
 
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

setup_ffmpeg()