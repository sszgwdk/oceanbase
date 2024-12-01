

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

    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    __m512 sum4 = _mm512_setzero_ps();
    __m512 sum5 = _mm512_setzero_ps();
    __m512 sum6 = _mm512_setzero_ps();
    __m512 sum7 = _mm512_setzero_ps();
    __m512 sum8 = _mm512_setzero_ps();

    // Fully unroll the loop to process 128 floats in total
    __m512 v11 = _mm512_load_ps(pVect1 + 0);
    __m512 v12 = _mm512_load_ps(pVect2 + 0);
    __m512 diff1 = _mm512_sub_ps(v11, v12);
    sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

    __m512 v21 = _mm512_load_ps(pVect1 + 16);
    __m512 v22 = _mm512_load_ps(pVect2 + 16);
    __m512 diff2 = _mm512_sub_ps(v21, v22);
    sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

    __m512 v31 = _mm512_load_ps(pVect1 + 32);
    __m512 v32 = _mm512_load_ps(pVect2 + 32);
    __m512 diff3 = _mm512_sub_ps(v31, v32);
    sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);

    __m512 v41 = _mm512_load_ps(pVect1 + 48);
    __m512 v42 = _mm512_load_ps(pVect2 + 48);
    __m512 diff4 = _mm512_sub_ps(v41, v42);
    sum4 = _mm512_fmadd_ps(diff4, diff4, sum4);

    __m512 v51 = _mm512_load_ps(pVect1 + 64);
    __m512 v52 = _mm512_load_ps(pVect2 + 64);
    __m512 diff5 = _mm512_sub_ps(v51, v52);
    sum5 = _mm512_fmadd_ps(diff5, diff5, sum5);

    __m512 v61 = _mm512_load_ps(pVect1 + 80);
    __m512 v62 = _mm512_load_ps(pVect2 + 80);
    __m512 diff6 = _mm512_sub_ps(v61, v62);
    sum6 = _mm512_fmadd_ps(diff6, diff6, sum6);

    __m512 v71 = _mm512_load_ps(pVect1 + 96);
    __m512 v72 = _mm512_load_ps(pVect2 + 96);
    __m512 diff7 = _mm512_sub_ps(v71, v72);
    sum7 = _mm512_fmadd_ps(diff7, diff7, sum7);

    __m512 v81 = _mm512_load_ps(pVect1 + 112);
    __m512 v82 = _mm512_load_ps(pVect2 + 112);
    __m512 diff8 = _mm512_sub_ps(v81, v82);
    sum8 = _mm512_fmadd_ps(diff8, diff8, sum8);

    // Horizontal add to get the final result
    sum1 = _mm512_add_ps(sum1, sum2);
    sum3 = _mm512_add_ps(sum3, sum4);
    sum5 = _mm512_add_ps(sum5, sum6);
    sum7 = _mm512_add_ps(sum7, sum8);

    sum1 = _mm512_add_ps(sum1, sum3);
    sum5 = _mm512_add_ps(sum5, sum7);

    sum1 = _mm512_add_ps(sum1, sum5);

    float res = _mm512_reduce_add_ps(sum1);

    return res;
}

float 
INT8L2Sqr512AVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    __mmask32 mask = 0xFFFFFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;

    size_t qty = *((size_t*)qty_ptr);
    uint32_t cTmp[16];

    uint8_t* pVect1 = (uint8_t*)pVect1v;
    uint8_t* pVect2 = (uint8_t*)pVect2v;
    const uint8_t* pEnd1 = pVect1 + qty;
    
    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        // 加载 32 个 int8_t 到 __m256i 寄存器
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2);

        // 计算差值
        __m256i diff = _mm256_sub_epi8(v1, v2);

        // 将差值扩展为 16 位整数
        __m512i diff_16 = _mm512_cvtepi8_epi16(diff);

        // 计算平方
        __m512i diff_sq = _mm512_mullo_epi16(diff_16, diff_16);

        // 累加结果
        sum512 = _mm512_add_epi32(sum512, diff_sq);

        // 移动指针
        pVect1 += 32;
        pVect2 += 32;
    }

    // 将累加结果从 AVX512 寄存器中提取出来
    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);

    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }

    return (float)res;
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

float
INT8InnerProduct512AVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    __mmask32 mask = 0xFFFFFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;
    size_t qty = *((size_t*)qty_ptr);
    int32_t cTmp[16];
    int8_t* pVect1 = (int8_t*)pVect1v;
    int8_t* pVect2 = (int8_t*)pVect2v;
    const int8_t* pEnd1 = pVect1 + qty;
    __m512i sum512 = _mm512_set1_epi32(0);
    while (pVect1 < pEnd1) {
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        pVect1 += 32;
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        pVect2 += 32;
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }
    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }
    return res;
}
float
INT8InnerProduct256AVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    __mmask16 mask = 0xFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;
    size_t qty = *((size_t*)qty_ptr);
    int32_t cTmp[16];
    int8_t* pVect1 = (int8_t*)pVect1v;
    int8_t* pVect2 = (int8_t*)pVect2v;
    const int8_t* pEnd1 = pVect1 + qty;
    __m512i sum512 = _mm512_set1_epi32(0);
    while (pVect1 < pEnd1) {
        __m128i v1 = _mm_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi32(v1);
        pVect1 += 16;
        __m128i v2 = _mm_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi32(v2);
        pVect2 += 16;
        sum512 = _mm512_add_epi32(sum512, _mm512_mullo_epi32(v1_512, v2_512));
    }
    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }
    return res;
}

}  // namespace vsag
