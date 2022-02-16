#!/bin/sh

docker build --no-cache -t tiskw/cufhe:`date +"%Y-%m-%d"` .

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
