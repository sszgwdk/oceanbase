#!/bin/bash

# 需要一个参数trace_id
if [ $# -ne 1 ]; then
    echo "Usage: $0 trace_id"
    exit 1
fi

trace_id=$1

store_dir=tmp
if [ ! -d $store_dir ]; then
    mkdir $store_dir
fi

datestr=$(date +%Y%m%d%H%M)
output_file="${store_dir}/${datestr}-${trace_id}.log"
vasg_file="${store_dir}/${datestr}-${trace_id}-vasg.log"

grep "$trace_id" /data/obcluster/log/* > $output_file
 
grep "Vsag" $output_file > $vasg_file

echo "grep success!"
echo "output file: $output_file"
echo "vasg file: $vasg_file"