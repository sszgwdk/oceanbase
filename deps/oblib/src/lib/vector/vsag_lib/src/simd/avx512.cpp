

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

float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
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

int32_t
L2SqrSQ4SIMD16ExtAVX512_dim128(const uint8_t *x, const uint8_t *y, int d) {
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi8(0xf);
    auto xx = _mm512_loadu_si512((__m512i *)(x));
    auto yy = _mm512_loadu_si512((__m512i *)(y));
    // 提取低 4 位和高 4 位
    __m512i xx1 = _mm512_and_si512(xx, mask);
    __m512i xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);
    __m512i yy1 = _mm512_and_si512(yy, mask);
    __m512i yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);
    // 计算差值并取绝对值
    __m512i d1 = _mm512_abs_epi8(_mm512_sub_epi8(xx1, yy1));
    __m512i d2 = _mm512_abs_epi8(_mm512_sub_epi8(xx2, yy2));

    // 将 8 位差值平方并累加到 16 位
    sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(d1, d1));
    sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(d2, d2));
    // 将 16 位整数转换为 32 位整数
    __m512i sum_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum, 0));
    sum_32 = _mm512_add_epi32(sum_32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum, 1)));

    // 使用 AVX512 归约指令直接求和
    return _mm512_reduce_add_epi32(sum_32);
}


}  // namespace vsag
