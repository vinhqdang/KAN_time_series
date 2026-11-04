"""List / extract single members of a remote zip via HTTP range requests (TLS verification on, through the
session proxy).  Used to fetch only the .csv members of large Causal Chambers zips that also contain images.
Usage: remote_zip.py list URL            -> member names and sizes
       remote_zip.py get URL MEMBER OUT  -> extract one member to OUT"""
import sys, io, zipfile, urllib.request, ssl

CTX = ssl.create_default_context()  # honours SSL_CERT_FILE; verification stays on

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=CTX) as r:
            self.size = int(r.headers["Content-Length"])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, context=CTX) as r:
            data = r.read()
        self.pos += len(data); return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

if __name__ == "__main__":
    f = io.BufferedReader(HTTPRangeFile(sys.argv[2]), buffer_size=1 << 20)
    z = zipfile.ZipFile(f)
    if sys.argv[1] == "list":
        for i in z.infolist():
            if not i.filename.lower().endswith((".png", ".jpg", ".jpeg", ".npy")):
                print(i.filename, i.file_size, i.compress_size)
        print("total members", len(z.infolist()))
    else:
        with z.open(sys.argv[3]) as src, open(sys.argv[4], "wb") as dst:
            while True:
                b = src.read(1 << 22)
                if not b: break
                dst.write(b)
        print("wrote", sys.argv[4])
