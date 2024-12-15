#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <map>
#include <sys/mman.h>
#include <random>
#include <cmath>

#include "../algorithm/hnswlib/hnswlib.h"
#include "../simd/simd.h"

// for serialize、deserialize
#include "../algorithm/hnswlib/stream_reader.h"
#include "../algorithm/hnswlib/stream_writer.h"

namespace vsag {


// 内存预取
inline void prefetch_L1(const void *address) {
#ifdef USE_SSE
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

inline void mem_prefetch(char *ptr, const int num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 28:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 27:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 26:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 25:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 24:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 23:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 22:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 21:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 20:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

// for 持久化、反序列化
template <typename T>
static void
GlassWriteOne(StreamWriter& writer, T& value) {
    writer.Write(reinterpret_cast<char*>(&value), sizeof(value));
}

template <typename T>
static void
GlassReadOne(StreamReader& reader, T& value) {
    reader.Read(reinterpret_cast<char*>(&value), sizeof(value));
}

template <typename T> struct align_alloc {
  T *ptr = nullptr;
  using value_type = T;
  T *allocate(int n) {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(T) + 63) >> 6 << 6;
      return ptr = (T *)std::aligned_alloc(64, sz);
    }
    int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
    ptr = (T *)std::aligned_alloc(1 << 21, sz);
    madvise(ptr, sz, MADV_HUGEPAGE);
    return ptr;
  }
  void deallocate(T *, int) { free(ptr); }
  template <typename U> struct rebind {
    typedef align_alloc<U> other;
  };
  bool operator!=(const align_alloc &rhs) { return ptr != rhs.ptr; }
};

inline void *alloc2M(size_t nbytes) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
  auto p = std::aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline void *alloc64B(size_t nbytes) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
  auto p = std::aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}


// 负责 0 层以上的边
struct HNSWInitializer {
  int N, K;
  int ep;
  std::vector<int> levels;
  std::vector<std::vector<int, align_alloc<int>>> lists;

  // wk: 预取
  int prefech_lines = 1;
  HNSWInitializer() = default;

  explicit HNSWInitializer(int n, int K = 0)
      : N(n), K(K), levels(n), lists(n) {
    if (K >= 16) {
      prefech_lines = K / 16;   // 16 * 4 = 64 正好是一个缓存行
    }
  }

  HNSWInitializer(const HNSWInitializer &rhs) = default;

  int at(int level, int u, int i) const {
    return lists[u][(level - 1) * K + i];
  }

  int &at(int level, int u, int i) { return lists[u][(level - 1) * K + i]; }

  const int *edges(int level, int u) const {
    return lists[u].data() + (level - 1) * K;
  }

  int *edges(int level, int u) { return lists[u].data() + (level - 1) * K; }


  template <typename Pool, typename Computer>
  void initialize(Pool &pool, const Computer &computer) const {
    int u = ep;
    auto cur_dist = computer(u);
    for (int level = levels[u]; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        // wk: 预取
        mem_prefetch((char *)edges(level, u), prefech_lines);
        const int *list = edges(level, u);
        for (int i = 0; i < K && list[i] != -1; ++i) {
          computer.prefetch(list[i], 1);
        }

        for (int i = 0; i < K && list[i] != -1; ++i) {
          int v = list[i];
          auto dist = computer(v);
          if (dist < cur_dist) {
            cur_dist = dist;
            u = v;
            changed = true;
          }
        }
      }
    }
    pool.insert(u, cur_dist);
    pool.vis.set(u);
  }

  void load(std::ifstream &reader) {
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    reader.read((char *)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur;
      reader.read((char *)&cur, 4);
      levels[i] = cur / K;
      lists[i].assign(cur, -1);
      reader.read((char *)lists[i].data(), cur * 4);
    }
  }

  // 适配 ob 反序列化
  void DeserializeImpl(StreamReader& reader) {
    GlassReadOne(reader, N);
    GlassReadOne(reader, K);
    GlassReadOne(reader, ep);
    if (K >= 16) {
      prefech_lines = K / 16;   // 16 * 4 = 64 正好是一个缓存行
    }
    for (int i = 0; i < N; ++i) {
      int cur;
      GlassReadOne(reader, cur);
      levels[i] = cur / K;
      // 非常关键，如果不做判断会导致空指针访问的异常
      if (cur > 0) {
        lists[i].assign(cur, -1);
        reader.Read((char *)lists[i].data(), cur * 4);
      }
    }
  }

  void Deserialize(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    DeserializeImpl(reader);
  }

  void save(std::ofstream &writer) const {
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)&ep, 4);
    for (int i = 0; i < N; ++i) {
      int cur = levels[i] * K;
      writer.write((char *)&cur, 4);
      writer.write((char *)lists[i].data(), cur * 4);
    }
  }
  // 适配 ob 序列化
  size_t calcSerializeSize() {
    size_t size = 0;
    size += 4;  // N
    size += 4;  // K
    size += 4;  // ep
    for (int i = 0; i < N; ++i) {
      int cur = levels[i] * K;
      size += 4;  // cur
      size += cur * 4;  // lists[i]
    }
    return size;
  }

  void SerializeImpl(StreamWriter& writer) {
    GlassWriteOne(writer, N);
    GlassWriteOne(writer, K);
    GlassWriteOne(writer, ep);
    for (int i = 0; i < N; ++i) {
      int cur = levels[i] * K;
      GlassWriteOne(writer, cur);
      // writer.Write((char *)(lists[i].data()), cur * 4);
      if (cur > 0) {
        // 很关键：只有当 cur > 0 时才会序列化，否则会crash，因为传入了空指针
        writer.Write((char *)lists[i].data(), cur * 4);
      }
    }
  }
  void Serialize(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    SerializeImpl(writer);
  }

  void Serialize(void* d) {
    char* dest = (char*)d;
    BufferStreamWriter writer(dest);
    SerializeImpl(writer);
  }

};



constexpr int EMPTY_ID = -1;

template <typename node_t> struct GlassGraph {
  int N, K;
  bool is_initialized = false;    // 判定是否已经初始化

  node_t *data = nullptr;

  int64_t* labels = nullptr; // 适配 ob 的 vids

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  // std::vector<int> eps;    // 不需要 eps

  GlassGraph() = default;

  // GlassGraph(node_t *edges, int N, int K) : N(N), K(K), data(edges) {}

  GlassGraph(int N, int K)
      : N(N), K(K), data((node_t *)alloc2M((size_t)N * K * sizeof(node_t))),
        labels((int64_t *)alloc2M((size_t)N * sizeof(int64_t)))  // 添加 label 
      {
        is_initialized = true;    // 判定是否已经初始化
      }

  // GlassGraph(const GlassGraph &g) : GlassGraph(g.N, g.K) {
  //   this->eps = g.eps;
  //   for (int i = 0; i < N; ++i) {
  //     for (int j = 0; j < K; ++j) {
  //       at(i, j) = g.at(i, j);
  //     }
  //   }
  //   if (g.initializer) {
  //     initializer = std::make_unique<HNSWInitializer>(*g.initializer);
  //   }
  // }

  bool initialized() const { return is_initialized; }

  void init(int N, int K) {
    // 已初始化
    if (is_initialized) {
      return;
    }
    is_initialized = true;

    data = (node_t *)alloc2M((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));

    // 为 labels 申请内存
    labels = (int64_t *)alloc2M((size_t)N * sizeof(int64_t));
    std::memset(labels, -1, N * sizeof(int64_t));

    this->K = K;
    this->N = N;
  }

  ~GlassGraph() { free(data); free(labels); } // 增加 free labels

  const int *edges(int u) const { return data + K * u; }

  int *edges(int u) { return data + K * u; }

  node_t at(int i, int j) const { return data[i * K + j]; }

  node_t &at(int i, int j) { return data[i * K + j]; }

  // 适配 ob 的 vids
  int64_t get_label(int u) const { return labels[u]; }
  void set_label(int u, int64_t vid) { labels[u] = vid; }

  void prefetch(int u, int lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    }
    // 不需要 eps 
    // else {
    //   for (auto ep : eps) {
    //     pool.insert(ep, computer(ep));
    //   }
    // }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    // int nep = eps.size();
    // writer.write((char *)&nep, 4);
    // writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
    printf("GlassGraph Saving done\n");
  }

  // 适配 ob 序列化
  size_t calcSerializeSize() {
    size_t size = 0;
    size += 4;  // N
    size += 4;  // K
    size += N * K * 4;  // data
    size += N * 8;  // labels
    if (initializer) {
      size += initializer->calcSerializeSize();
    }
    return size;
  }

  void SerializeImpl(StreamWriter& writer) {
    static_assert(std::is_same_v<node_t, int32_t>);
    GlassWriteOne(writer, N);
    GlassWriteOne(writer, K);
    writer.Write((char *)data, N * K * 4);
    // labels 序列化
    writer.Write((char *)labels, N * 8);
    if (initializer) {
      initializer->SerializeImpl(writer);
    }
  }
  // just for test
  void test_SerializeImpl(StreamWriter& writer) {
    static_assert(std::is_same_v<node_t, int32_t>);
    GlassWriteOne(writer, N);
    GlassWriteOne(writer, K);
    writer.Write((char *)data, N * K * 4);
    // labels 序列化
    writer.Write((char *)labels, N * 8);
    if (initializer) {
      initializer->SerializeImpl(writer);
    }
  }

  void Serialize(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    // SerializeImpl(writer);
    SerializeImpl(writer);
  }

  void Serialize(void* d) {
    char* dest = (char*)d;
    BufferStreamWriter writer(dest);
    SerializeImpl(writer);
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    // int nep;
    // reader.read((char *)&nep, 4);
    // eps.resize(nep);
    // reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    printf("GlassGraph Loding done\n");
  }

  // 适配 ob 反序列化
  void DeserializeImpl(StreamReader& reader) {
    static_assert(std::is_same_v<node_t, int32_t>);
    GlassReadOne(reader, N);
    GlassReadOne(reader, K);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.Read((char *)data, N * K * 4);
    // labels 反序列化
    labels = (int64_t *)alloc2M((size_t)N * 8);
    reader.Read((char *)labels, N * 8);
    // initializer 反序列化
    initializer = std::make_unique<HNSWInitializer>(N);
    initializer->DeserializeImpl(reader);
    is_initialized = true;  // 完成初始化
  }

  void Deserialize(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    DeserializeImpl(reader);
  }
};

template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};


template <typename Block = uint64_t> struct GlassBitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int nbytes;
  Block *data;
  explicit GlassBitset(int n)
      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((uint64_t *)alloc64B(nbytes)) {
    memset(data, 0, nbytes);
  }
  ~GlassBitset() { free(data); }
  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }
  bool get(int i) {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }

  void *block_address(int i) { return data + i / block_size; }
};


template <typename dist_t> struct LinearPool {
  LinearPool(int n, int capacity, int = 0)
      : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
  GlassBitset<uint64_t> vis;
};

inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}
struct SearcherBase {
  // virtual void SetData(const float *data, int n, int dim) = 0;
  // virtual void Optimize(int num_threads = 0) = 0;  // 暂时不需要 Optimize
  virtual void Search(const float *q, int k, int *dst) const = 0;

  virtual std::vector<std::pair<int64_t, float>> Search(const float *q, int k) const = 0; // 增加一个接口
  virtual void SetEf(int ef) = 0;
  virtual ~SearcherBase() = default;
};

template <typename Quantizer> struct Searcher : public SearcherBase {

  int d;
  int nb;
  GlassGraph<int>* graph;
  Quantizer* quant;

  // Search parameters
  int ef = 120;

  // Memory prefetch parameters
  int po = 1;
  int pl = 1;

  // Optimization parameters
  constexpr static int kOptimizePoints = 1000;
  constexpr static int kTryPos = 10;
  constexpr static int kTryPls = 5;
  constexpr static int kTryK = 10;
  int sample_points_num;
  std::vector<float> optimize_queries;
  const int graph_po;

  // 更改构造函数
  // Searcher(const GlassGraph<int> &graph) : graph(graph), graph_po(graph.K / 16) {}
  Searcher(GlassGraph<int> *graph, Quantizer *quant)
      : graph(graph), quant(quant), graph_po(graph->K / 16) {
    // 不走 set_data，所以这里需要初始化
    this->nb = graph->N;
    this->d = quant->d;
    // 此时 graph 和 quant 都已经完成构建
  }

  // 这里会报错mad，暂时用不到
  // void SetData(const float *data, int n, int dim) override {
  //   this->nb = n;
  //   this->d = dim;
  //   quant = Quantizer(d);
  //   // quant.train(data, n);
  //   quant->train(data, n);

  //   // wk: 用于 Optimize 预取的参数
  //   // sample_points_num = std::min(kOptimizePoints, nb - 1);
  //   // std::vector<int> sample_points(sample_points_num);
  //   // std::mt19937 rng;
  //   // GenRandom(rng, sample_points.data(), sample_points_num, nb);
  //   // optimize_queries.resize(sample_points_num * d);
  //   // for (int i = 0; i < sample_points_num; ++i) {
  //   //   memcpy(optimize_queries.data() + i * d, data + sample_points[i] * d,
  //   //          d * sizeof(float));
  //   // }
  // }

  void SetEf(int ef) override { this->ef = ef; }

//   void Optimize(int num_threads = 0) override {
//     if (num_threads == 0) {
//       num_threads = std::thread::hardware_concurrency();
//     }
//     std::vector<int> try_pos(std::min(kTryPos, graph.K));
//     std::vector<int> try_pls(
//         std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
//     std::iota(try_pos.begin(), try_pos.end(), 1);
//     std::iota(try_pls.begin(), try_pls.end(), 1);
//     std::vector<int> dummy_dst(kTryK);
//     printf("=============Start optimization=============\n");
//     { // warmup
// #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
//       for (int i = 0; i < sample_points_num; ++i) {
//         Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
//       }
//     }

//     float min_ela = std::numeric_limits<float>::max();
//     int best_po = 0, best_pl = 0;
//     for (auto try_po : try_pos) {
//       for (auto try_pl : try_pls) {
//         this->po = try_po;
//         this->pl = try_pl;
//         auto st = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
//         for (int i = 0; i < sample_points_num; ++i) {
//           Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
//         }

//         auto ed = std::chrono::high_resolution_clock::now();
//         auto ela = std::chrono::duration<double>(ed - st).count();
//         if (ela < min_ela) {
//           min_ela = ela;
//           best_po = try_po;
//           best_pl = try_pl;
//         }
//       }
//     }
//     this->po = 1;
//     this->pl = 1;
//     auto st = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
//     for (int i = 0; i < sample_points_num; ++i) {
//       Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
//     }
//     auto ed = std::chrono::high_resolution_clock::now();
//     float baseline_ela = std::chrono::duration<double>(ed - st).count();
//     printf("settint best po = %d, best pl = %d\n"
//            "gaining %.2f%% performance improvement\n============="
//            "Done optimization=============\n",
//            best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
//     this->po = best_po;
//     this->pl = best_pl;
//   }

  void Search(const float *q, int k, int *dst) const override {
    auto computer = quant->get_computer(q);
    // searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
    LinearPool<typename Quantizer::Computer::dist_type>
        pool(nb, std::max(k, ef), k);
    graph->initialize_search(pool, computer);
    SearchImpl(pool, computer);
    quant->reorder(pool, q, dst, k);
  }

  // 新增接口：返回 <label, dist>
  std::vector<std::pair<int64_t, float>> Search(const float *q, int k) const override {
    auto computer = quant->get_computer(q);
    // searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
    LinearPool<typename Quantizer::Computer::dist_type>
        pool(nb, std::max(k, ef), k);
    graph->initialize_search(pool, computer);
    SearchImpl(pool, computer);
    std::vector<std::pair<int, float>> results_ids = quant->reorder(pool, q, k);   // 使用新增的reorder接口
    // 转换为原始的 label
    std::vector<std::pair<int64_t, float>> results_labels;
    // for (int i = 0; i < results_ids.size(); i++) {
    for (int i = results_ids.size() - 1; i >= 0; i--) { // 倒序
      results_labels.emplace_back(graph->get_label(results_ids[i].first), results_ids[i].second);
    }
    return std::move(results_labels);
  }

  // TODO(wk): searchwithFilter
  void SearchWithFilter(const float *q, int k, int *dst, const int *filter);

  template <typename Pool, typename Computer>
  void SearchImpl(Pool &pool, const Computer &computer) const {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph->prefetch(u, graph_po);
      // for (int i = 0; i < po; ++i) {
      // 最大化预取，pl = 1，每次取64字节，正好是一条量化向量数据
      for (int i = 0; i < graph->K; ++i) {
        int to = graph->at(u, i);
        if (to == -1) {
          break;
        }
        // 已经访问过的也不进行预取
        if (pool.vis.get(to)) {
          continue;
        }        
        computer.prefetch(to, pl);
      }
      for (int i = 0; i < graph->K; ++i) {
        int v = graph->at(u, i);
        if (v == -1) {
          break;
        }
        // 这个预取可以省去，上面预取完了
        // if (i + po < graph->K && graph->at(u, i + po) != -1) {
        //   int to = graph->at(u, i + po);
        //   computer.prefetch(to, pl);
        // }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
      }
    }
  }
};

// 量化器

enum class Metric {
  L2,
  IP,
};

inline constexpr int64_t do_align(int64_t x, int64_t align) {
  return (x + align - 1) / align * align;
}

// 用于存原始数据
// template <Metric metric, int DIM = 0> struct FP32Quantizer {
template <Metric metric>
struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;

  size_t d;     // d 由 int -> size_t
  int d_align;
  int64_t code_size;      // 前面都在初始化时就能得到

  int cur_element_count;    // 统计当前元素个数
  char *codes = nullptr;

  // FP32Quantizer() = default;  // 删除默认构造函数

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {
        cur_element_count = 0;    // 初始化为0
        // 预先分配空间，不动态分配
        // codes 里边存所有的原始向量数据
        codes = (char *)alloc2M(code_size * 1000010);
      }

  ~FP32Quantizer() { free(codes); }

  void train(const float *data, int64_t n) {
    codes = (char *)alloc2M(n * code_size);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  // 适配 ob add_index，注意内存已经预先分配好了
  void add_points(const float *data, int n) {
    if (codes == nullptr) {
      throw std::runtime_error("FP32Quantizer: codes is nullptr");
    }

    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(cur_element_count + i));
    }
    cur_element_count += n;
  }

  void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

  char *get_data(int u) const { return codes + u * code_size; }

  template <typename Pool>
  void reorder(const Pool &pool, const float *, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  // template <int DALIGN = do_align(DIM, kAlign)> 
  struct Computer {
    using dist_type = float;
    // constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
    DistanceFunc dist_func; // change for distance
    const FP32Quantizer &quant;
    float *q = nullptr;
    Computer(const FP32Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)) {
      std::memcpy(q, query, quant.d * 4);

      // dist_func = metric == Metric::L2 ? GetL2DistanceFunc(quant.d) : GetIPDistanceFunc(quant.d); // change for distance
      dist_func = metric == Metric::L2 ? GetL2DistanceFunc(quant.d) : GetInnerProductDistanceFunc(quant.d); // change for distance
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), &quant.d);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    // return Computer<0>(*this, query);
    return Computer(*this, query);
  }

  // int d, d_align;
  // int cur_element_count;    // 统计当前元素个数
  // int64_t code_size;
  // char *codes = nullptr;

  // 适配 ob 序列化
  size_t calcSerializeSize() {
    size_t size = 0;
    size += sizeof(int); // cur_element_count
    size += code_size * cur_element_count;
    return size;
  }

  void SerializeImpl(StreamWriter& writer) {
    GlassWriteOne(writer, cur_element_count);
    writer.Write((char *)codes, code_size * cur_element_count);
  }
  void Serialize(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    SerializeImpl(writer);
  }

  void Serialize(void* d) {
    char* dest = (char*)d;
    BufferStreamWriter writer(dest);
    SerializeImpl(writer);
  }

  // 适配 ob 反序列化
  void DeserializeImpl(StreamReader& reader) {
    GlassReadOne(reader, cur_element_count);
    // 已经预先分配好了内存
    // codes = (char *)alloc2M(code_size * cur_element_count);
    reader.Read((char *)codes, code_size * cur_element_count);
  }

  void Deserialize(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    DeserializeImpl(reader);
  }

};


template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++size);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return pool[size].id;
  }
  
  // 增加一个接口，用于同时获取 id 和 dist
  std::pair<int, float> pop_pair() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    std::pair<int, float> result = {pool[size].id, pool[size].distance};
    return result;
  }

  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

// template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
//           int DIM = 0>
struct SQ4Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 128;
  float mx = -HUGE_VALF, mi = HUGE_VALF, dif;
  int d, d_align;

  int64_t code_size;

  int cur_element_count;    // 统计当前元素个数
  data_type *codes = nullptr;

  // Reorderer reorderer;
  FP32Quantizer<Metric::L2> reorderer;

  // SQ4Quantizer() = default;  // 删除默认构造函数

  explicit SQ4Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align / 2),
        reorderer(dim) {
          cur_element_count = 0;    // 初始化为0
        }

  ~SQ4Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    codes = (data_type *)alloc2M(n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  // 适配 ob add_index
  void add_points(const float *data, int n) {
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    cur_element_count += n;
    // 不进行 encode ，还没训练完成
    // 将原始数据存到 reorderer 中
    reorderer.add_points(data, n);
  }

  // 在训练完成后，把 reorderer 中存的原始数据提出来并进行编码
  void encode_all_points() {
    if (cur_element_count != reorderer.cur_element_count) {
      throw std::runtime_error("sq4 quantizer's cur_element_count is not equal to reorderer's cur_element_count");
    }
    codes = (data_type *)alloc2M(cur_element_count * code_size);  // 为codes分配空间

    for (int i = 0; i < cur_element_count; ++i) {
      encode((const float *)reorderer.get_data(i), get_data(i));
    }
  }

  char *get_data(int u) const { return (char *)codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi) / dif;
      if (x < 0.0) {
        x = 0.0;
      }
      if (x > 0.999) {
        x = 0.999;
      }
      uint8_t y = 16 * x;
      if (j & 1) {
        to[j / 2] |= y << 4;
      } else {
        to[j / 2] |= y;
      }
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float *q, int *dst, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    // searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
    MaxHeap<float> heap(k);
    for (int i = 0; i < cap; ++i) {
      if (i + 1 < cap) {
        computer.prefetch(pool.id(i + 1), 1);
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      dst[i] = heap.pop();
    }
  }

  // 增加一个 reorder 接口，用于返回vector<int>
  template <typename Pool>
  std::vector<std::pair<int, float>> reorder(const Pool &pool, const float *q, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    // searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
    MaxHeap<float> heap(k);

    // wk: 提前预取 4 个向量
    if (cap > 0) {
      computer.prefetch(pool.id(0), 8);
      if (cap > 1) {
        computer.prefetch(pool.id(1), 8);
      }
      if (cap > 2) {
        computer.prefetch(pool.id(2), 8);
      }
      if (cap > 3) {
        computer.prefetch(pool.id(3), 8);
      }
    }

    for (int i = 0; i < cap; ++i) {
      if (i + 4 < cap) {
        // computer.prefetch(pool.id(i + 1), 1);
        computer.prefetch(pool.id(i + 4), 8); // 预取优化：128 * 4 = 64 * 8，一个向量就要预取8条缓存
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }

    std::vector<std::pair<int, float>> result;

    for (int i = 0; i < k; ++i) {
      // wk: 这里不能乱改
    // auto heap_size = heap.size;   // 可能堆中的元素个数小于k
    // float min_dist;
    // for (int i = 0; i < std::min(k, heap_size); ++i) {
      // dst[i] = heap.pop();
      result.push_back(heap.pop_pair());
    }
    return std::move(result);
  }

  // template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
  struct Computer {
    using dist_type = int32_t;
    // constexpr static auto dist_func = L2SqrSQ4;
    SQ4DistanceFunc dist_func; // change for distance
    const SQ4Quantizer &quant;
    uint8_t *q;
    Computer(const SQ4Quantizer &quant, const float *query)
        : quant(quant), q((uint8_t *)alloc64B(quant.code_size)) {
      quant.encode(query, (char *)q);

      dist_func = GetSQ4L2DistanceFunc(quant.d); // change for distance
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    // return Computer<0>(*this, query);
    return Computer(*this, query);
  }

  // 适配 ob 序列化
  size_t calcSerializeSize() {
    size_t size = 0;
    size += sizeof(float);  // mx
    size += sizeof(float);  // mi
    size += sizeof(float);  // dif
    size += sizeof(int);    // cur_element_count
    size += code_size * cur_element_count;  // codes size
    size += reorderer.calcSerializeSize();  // reorderer
    return size;
  }
  void SerializeImpl(StreamWriter& writer) {
    GlassWriteOne(writer, mx);
    GlassWriteOne(writer, mi);
    GlassWriteOne(writer, dif);
    GlassWriteOne(writer, cur_element_count);
    writer.Write((char *)codes, code_size * cur_element_count);
    // 序列化 reorderer
    reorderer.SerializeImpl(writer);
  }
  void Serialize(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    SerializeImpl(writer);
  }
  void Serialize(void* d) {
    char* dest = (char*)d;
    BufferStreamWriter writer(dest);
    SerializeImpl(writer);
  }

  // 适配 ob 反序列化
  void DeserializeImpl(StreamReader& reader) {
    GlassReadOne(reader, mx);
    GlassReadOne(reader, mi);
    GlassReadOne(reader, dif);
    GlassReadOne(reader, cur_element_count);
    codes = (data_type *)alloc2M(code_size * cur_element_count);    // 需要分配内存
    reader.Read((char *)codes, code_size * cur_element_count);
    // 反序列化 reorderer
    reorderer.DeserializeImpl(reader);
  }

  void Deserialize(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    DeserializeImpl(reader);
  }

};


} // namespace vsag:glass