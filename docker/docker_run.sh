#!/bin/sh

docker run -itd -u `id -u`:`id -g` -v `pwd`:/workspace -w /workspace --name cufhe --entrypoint /bin/sh --gpus all tiskw/cufhe:2021-02-10

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
