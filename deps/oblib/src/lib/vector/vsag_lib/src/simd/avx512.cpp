

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
    // size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    
    __m512 diff1, v11, v12, diff2, v21, v22, diff3, v31, v32, diff4, v41, v42, diff5, v51, v52, diff6, v61, v62, diff7, v71, v72, diff8, v81, v82;
    __m512 sum = _mm512_set1_ps(0);
    v11 = _mm512_loadu_ps(pVect1);
    v12 = _mm512_loadu_ps(pVect2);
    diff1 = _mm512_sub_ps(v11, v12);
    sum = _mm512_fmadd_ps(diff1, diff1, sum);
    v21 = _mm512_loadu_ps(pVect1 + 16);
    v22 = _mm512_loadu_ps(pVect2 + 16);
    diff2 = _mm512_sub_ps(v21, v22);
    sum = _mm512_fmadd_ps(diff2, diff2, sum);
    v31 = _mm512_loadu_ps(pVect1 + 32);
    v32 = _mm512_loadu_ps(pVect2 + 32);
    diff3 = _mm512_sub_ps(v31, v32);
    sum = _mm512_fmadd_ps(diff3, diff3, sum);
    v41 = _mm512_loadu_ps(pVect1 + 48);
    v42 = _mm512_loadu_ps(pVect2 + 48);
    diff4 = _mm512_sub_ps(v41, v42);
    sum = _mm512_fmadd_ps(diff4, diff4, sum);
    v51 = _mm512_loadu_ps(pVect1 + 64);
    v52 = _mm512_loadu_ps(pVect2 + 64);
    diff5 = _mm512_sub_ps(v51, v52);
    sum = _mm512_fmadd_ps(diff5, diff5, sum);
    v61 = _mm512_loadu_ps(pVect1 + 80);
    v62 = _mm512_loadu_ps(pVect2 + 80);
    diff6 = _mm512_sub_ps(v61, v62);
    sum = _mm512_fmadd_ps(diff6, diff6, sum);
    v71 = _mm512_loadu_ps(pVect1 + 96);
    v72 = _mm512_loadu_ps(pVect2 + 96);
    diff7 = _mm512_sub_ps(v71, v72);
    sum = _mm512_fmadd_ps(diff7, diff7, sum);
    v81 = _mm512_loadu_ps(pVect1 + 112);
    v82 = _mm512_loadu_ps(pVect2 + 112);
    diff8 = _mm512_sub_ps(v81, v82);
    sum = _mm512_fmadd_ps(diff8, diff8, sum);

    // 原来的代码
    // size_t qty16 = qty >> 4;

    // const float* pEnd1 = pVect1 + (qty16 << 4);

    // __m512 diff, v1, v2;
    // // whp 
    // __m512 sum = _mm512_set1_ps(0);

    // while (pVect1 < pEnd1) {
    //     v1 = _mm512_loadu_ps(pVect1);
    //     pVect1 += 16;
    //     v2 = _mm512_loadu_ps(pVect2);
    //     pVect2 += 16;
    //     diff = _mm512_sub_ps(v1, v2);
    //     // whp
    //     sum = _mm512_fmadd_ps(diff, diff, sum);
    //     // sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    // }

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

}  // namespace vsag
