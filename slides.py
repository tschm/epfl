#!/usr/bin/python
import glob
import os
import subprocess


def mkdir(directory):
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)


mkdir(directory="pdf")

for f in glob.glob(os.path.join("work", "*.ipynb")):
    file = f.lstrip("work")
    url = "http://localhost:{PORT}/notebooks{file}".format(PORT=8888, file=file)
    out = "/slides{file}.pdf".format(file=file.rstrip(".ipynb"))
    subprocess.check_call('docker run --rm -t -v `pwd`/pdf:/slides --net=host astefanutti/decktape:2.9 rise {url} {out}'.format(url=url, out=out), shell=True)
