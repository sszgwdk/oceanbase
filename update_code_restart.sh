#!/bin/bash

# 重新编译 + 切换二进制文件
bash build.sh release --make -j2
cp ./build_release/deps/oblib/src/lib/vector/vsag_lib/src/simd/libsimd.a ./deps/3rd/usr/local/oceanbase/deps/devel/lib/vsag_lib/libsimd.a

# 切换二进制文件
./tools/deploy/obd.sh stop -n obcluster
cp build_release/src/observer/observer /data/obcluster/bin/observer
echo "success cp observer to /data/obcluster/bin/observer"
sleep 3

./tools/deploy/obd.sh start -n obcluster

echo "success restart observer"
echo "start obclient"
sleep 3
./deps/3rd/u01/obclient/bin/obclient -h 127.0.0.1 -P2881 -uroot

# perf 火焰图
# ps -aux | grep observer

# sudo perf record -p 1414024 -g -F 99 --call-graph dwarf -- sleep 10
# sudo perf record -p 1414024 -g --call-graph dwarf -- sleep 8
# #本地调试

# create database test;
# use test;
# alter system set ob_vector_memory_limit_percentage=34;
# set ob_log_level=debug;
# # 验证日志级别
# show variables like "%ob_log_level%";

# # 建表和初始化数据
# DROP TABLE IF EXISTS t1;
# CREATE TABLE t1 (id int,c1 int, embedding vector(8), primary key(id),key(c1));
# insert into t1 values (1,1, '[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]');
# insert into t1 values (2,2, '[3.0,2.0,4.0,4.0,5.0,6.0,7.0,8.0]');
# insert into t1 values (3,3, '[3.0,2.0,4.0,4.0,5.0,6.0,7.0,7.0]');
# # 创建索引
# create vector index idx1 on t1(embedding) with (distance=L2, type=hnsw, lib=vsag, m=16, ef_construction=200);

# ALTER SYSTEM MAJOR FREEZE;

# # query 注意只有开始查询才会真正创建索引
# SELECT * FROM t1 ORDER BY l2_distance(embedding, '[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]') APPROXIMATE LIMIT 10;
# select last_trace_id();