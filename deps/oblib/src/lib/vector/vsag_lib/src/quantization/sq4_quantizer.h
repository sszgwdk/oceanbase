
// // Copyright 2024-present the vsag project
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #pragma once

// #include <cstring>
// #include <limits>
// #include <vector>

// //#include "simd/sq4_simd.h"

// // 按照新vsag写 sq4。不用模板，传进来的数据是float，经过encoder变成uint8
// // decoder 返回uint8/float
// namespace vsag {
// class SQ4Quantizer {
// public:
//     explicit SQ4Quantizer(int dim);

//     bool EncodeOneImpl(const float* data, uint8_t* codes) const;

//     bool EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

//     bool DecodeOneImpl(const uint8_t* codes, DataType* data);

//     bool DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);
// private:
//     uint64_t dim_{0}; // 原数据维度
//     std::vector<float> lower_bound_{};
//     std::vector<float> diff_{};
//     Allocator* const allocator_{nullptr};
// };


// SQ4Quantizer::SQ4Quantizer(int dim, Allocator* allocator){
//     dim_ = 128;
//     lower_bound_.resize(dim);
//     diff_.resize(dim);
// }

// //code[0] = data[0] | data[1] << 4
// // 高4位放原数据奇数维，低4位放偶数维

// bool SQ4Quantizer::EncodeOneImpl(const float* data, uint8_t* codes) const {
//     float delta = 0;
//     uint8_t scaled = 0;
//     const float* cur = data;

//     for (uint64_t d = 0; d < this->dim_; d++) {
//         delta = 1.0f * (cur[d] - lower_bound_[d]) / diff_[d];
//         if (delta < 0.0) {
//             delta = 0;
//         } else if (delta > 0.999) {
//             delta = 1;
//         }
//         scaled = 15 * delta;

//         if (d & 1) {
//             codes[d >> 1] |= scaled << 4;
//         } else {
//             codes[d >> 1] = 0;
//             codes[d >> 1] |= scaled;
//         }
//     }
//     return true;
// }


// bool SQ4Quantizer::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
//     for (uint64_t i = 0; i < count; ++i) {
//         this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
//     }
//     return true;
// }

// // 解码成float向量
// bool SQ4Quantizer DecodeOneFloat(const uint8_t* codes, float* data) {
//     for (uint64_t d = 0; d < this->dim_; d++) {
//         if (d & 1) {
//             data[d] = ((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff_[d] + lower_bound_[d];
//         } else {
//             data[d] = (codes[d >> 1] & 0x0f) / 15.0 * diff_[d] + lower_bound_[d];
//         }
//     }
//     return true;
// }

// // 解码成uint8向量
// bool SQ4Quantizer Decodeuint8(const uint8_t* codes, uint8_t* data) {
//     for (uint64_t d = 0; d < this->dim_; d++) {
//         // todo：解码是否需要+0.5
//         if (d & 1) {
//             data[d] = (uint8_t)(((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff_[d] + lower_bound_[d]);
//         } else {
//             data[d] = (uint8_t)((codes[d >> 1] & 0x0f) / 15.0 * diff_[d] + lower_bound_[d]);
//         }
//     }
//     return true;
// }

// // 应该不调用
// bool SQ4Quantizer::DecodeBatchuint8(const uint8_t* codes, uint8_t* data, uint64_t count) {
//     for (uint64_t i = 0; i < count; ++i) {
//         this->Decodeuint8(codes + i * this->code_size_, data + i * this->dim_);
//     }
//     return true;
// }

// // 编码之后距离计算
// inline float SQ4Quantizer::SQ4ComputeL2Sqr(const uint8_t* codes1, const uint8_t* codes2) {
//     return SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
// }





// }  // namespace vsag
