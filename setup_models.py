"""Download the two required models, verify SHA256, never upload video."""
from pathlib import Path
import hashlib
import urllib.request
ROOT=Path(__file__).resolve().parent/'models'
MODELS=[
 ('shuttle_yolo11s.pt','https://huggingface.co/Huydinh1205/shuttle_yolo11s/resolve/main/shuttle_yolo11s.pt','fa978a0cb44f7230c44a4d8c6d196e471b2a2f274f9119c39e1bcc8dcf5e6d7c'),
 ('yolo11n-pose.pt','https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt','869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0')]
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,url,expected in MODELS:
        target=ROOT/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
            print(name,'verified');continue
        tmp=target.with_suffix('.download')
        try:
            with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
            if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'{name}: checksum mismatch; refusing to use changed model')
            tmp.replace(target);print(name,'downloaded and verified')
        finally:tmp.unlink(missing_ok=True)
if __name__=='__main__':main()
