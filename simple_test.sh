./deps/3rd/u01/obclient/bin/obclient -h 127.0.0.1 -P2881 -uroot@perf
use test;
alter system set ob_vector_memory_limit_percentage=34;
set ob_log_level=debug;
# 验证日志级别
show variables like "%ob_log_level%";

# 建表和初始化数据
DROP TABLE IF EXISTS t1;
CREATE TABLE t1 (id int,c1 int, embedding vector(8), primary key(id),key(c1));
insert into t1 values (1,1, '[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]');
insert into t1 values (2,2, '[0.1,0.4,0.9,0.2,0.5,0.8,0.3,0.6]');
insert into t1 values (3,3, '[0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]');
insert into t1 values (4,4, '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]');
insert into t1 values (5,5, '[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]');
insert into t1 values (6,6, '[0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.0]');
insert into t1 values (7,7, '[0.5,0.6,0.7,0.8,0.9,0.0,0.1,0.2]');
insert into t1 values (8,8, '[0.7,0.8,0.9,0.0,0.1,0.2,0.3,0.4]');
insert into t1 values (9,9, '[0.9,0.0,0.1,0.2,0.3,0.4,0.5,0.6]');

# 创建索引
create vector index idx1 on t1(embedding) with (distance=L2, type=hnsw, lib=vsag, m=64, ef_construction=200);
ALTER SYSTEM MAJOR FREEZE;

# query 注意只有开始查询才会真正创建索引
SELECT id FROM t1 ORDER BY l2_distance(embedding, '[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]') APPROXIMATE LIMIT 10;
select last_trace_id();

# 想要反复测直接drop表就可以清空数据了e
drop table t1;