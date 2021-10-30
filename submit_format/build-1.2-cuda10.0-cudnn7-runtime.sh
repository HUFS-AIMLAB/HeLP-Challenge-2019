#! /bin/bash

dir=$(dirname "$0")

docker build -t csv_1 -f $dir/Dockerfile $dir
