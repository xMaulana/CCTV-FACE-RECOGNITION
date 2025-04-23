import threading
import cv2
from .tools import get_ms, sleep_ms
import numpy as np
import subprocess

class RTSPStream:
    def __init__(self, url:str|int, dev:bool=False, recon_interv:int=2000, api_reference:int=None):
        if not dev:
            assert url.startswith("rtsp")
            
        self.url = url
        self.recon_interv = recon_interv
        self.cap = None
        self.frame = None
        self.temp_frame = None
        self.stopped = False
        self.api_reference = api_reference
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print(f"Reconnecting | {self.url}")
                    self.connect()

                ret, frame = self.cap.read()

                if ret:
                    with self.lock:
                        self.frame = frame
                        self.temp_frame = self.frame
                else:
                    print(f"Koneksi terputus | {self.url}")
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                break

    def connect(self):
        if self.api_reference is not None:
            self.cap = cv2.VideoCapture(self.url, self.api_reference)
        else:
            self.cap = cv2.VideoCapture(self.url)

        if self.cap.isOpened():
            print(f"Koneksi sukses | {self.url}")
        else:
            print(f"Koneksi gagal | {self.url}")

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else self.temp_frame
        
    def stop(self):
        self.stopped = True

        if self.cap is not None:
            self.cap.release()

        print(f"Streaming dihentikan | {self.url}")

    def get_fromcv(self, wg:int):
        ret = None
        while True:
            if self.cap is not None:
                with self.lock:
                    ret = self.cap.get(wg)
                break
        return ret
    
class RTSP_Server:
    def __init__(self, url:str, channel:int, vidsize:list|set= (720,480), fps:int=25):
        assert url is not None or channel is not None
        url = url.rstrip("/")
        self.stream_path = url + "/" + str(channel)
        self.vidsize = vidsize
        self.fps = fps
        self.process = None

        self.ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.vidsize[0]}x{self.vidsize[1]}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-crf', '1',
            '-b:v', '3M',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'rtsp',
            self.stream_path
        ]
    
    def start(self):
        self.process = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"STARTING RTSP | {self.stream_path}")
        return self
    
    def stop(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            print(f"{self.stream_path} dihentikan")

    def write_frame(self, frame:cv2.typing.MatLike):
        if self.process and self.process.stdin:
            try:
                frame = cv2.resize(frame, self.vidsize, interpolation=cv2.INTER_LINEAR)
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print(f"{self.stream_path} berhenti")


def compress_frame(frame:np.ndarray, quality=80):
    frame = frame.astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    _, buffer = cv2.imencode(".webp", frame, encode_param)

    return buffer.tobytes()