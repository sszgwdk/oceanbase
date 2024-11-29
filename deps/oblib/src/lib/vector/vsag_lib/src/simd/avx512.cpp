

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <x86intrin.h>

#include <iostream>

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

// float L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     float* pVect1 = (float*)pVect1v;
//     float* pVect2 = (float*)pVect2v;
//     size_t qty = *((size_t*)qty_ptr);
//     float PORTABLE_ALIGN64 TmpRes[16];
//     size_t qty16 = qty >> 4;

//     const float* pEnd1 = pVect1 + (qty16 << 4);

//     __m512 diff, v1, v2;
//     __m512 sum = _mm512_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         v1 = _mm512_loadu_ps(pVect1);
//         pVect1 += 16;
//         v2 = _mm512_loadu_ps(pVect2);
//         pVect2 += 16;
//         diff = _mm512_sub_ps(v1, v2);
//         // sum = _mm512_fmadd_ps(diff, diff, sum);
//         sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
//     }

//     _mm512_store_ps(TmpRes, sum);
//     float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
//                 TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
//                 TmpRes[13] + TmpRes[14] + TmpRes[15];

//     return (res);
// }


void floatVectorToIntVector(const void* fv, int* iv) {
    float* floatPtr = (float*)fv;
    for (int i = 0; i < 128; i++) {
        iv[i] = (int)floatPtr[i];
    }
}
float L2SqrSIMD16ExtAVX512(const void* pVect1, const void* pVect2, const void* qty_ptr) {

    __m512 sum = _mm512_setzero_ps();  // 初始化累加和为零

    int* iv1  = (int*)malloc(128 * sizeof(int));
    int* iv2  = (int*)malloc(128 * sizeof(int));
    floatVectorToIntVector(pVect1, iv1);
    floatVectorToIntVector(pVect1, iv2);

    for (int i = 0; i < 128; i += 16) {
        // 加载16个整数
        __m512i v1 = _mm512_loadu_si512((__m512i*)(iv1 + i));
        __m512i v2 = _mm512_loadu_si512((__m512i*)(iv2 + i));

        // 计算差值
        __m512i diff = _mm512_sub_epi32(v1, v2);

        // 计算差值的平方
        __m512i diff_sq = _mm512_mullo_epi32(diff, diff);

        // 累加平方值
        sum = _mm512_add_epi32(sum, diff_sq);
    }
    __m512i sum_h1 = _mm512_add_epi32(sum, _mm512_permutexvar_epi32(_mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7), sum));
    __m512i sum_h2 = _mm512_add_epi32(sum_h1, _mm512_permutexvar_epi32(_mm512_set_epi32(4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3), sum_h1));
    __m512i sum_h3 = _mm512_add_epi32(sum_h2, _mm512_permutexvar_epi32(_mm512_set_epi32(2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1), sum_h2));
    __m512i sum_h4 = _mm512_add_epi32(sum_h3, _mm512_permutexvar_epi32(_mm512_set_epi32(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0), sum_h3));

    // 提取最终结果
    uint32_t result[16];
    _mm512_storeu_si512((__m512i*)result, sum_h4);
    uint64_t final_result = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];

    return (float)final_result;
}


float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

}  // namespace vsag
