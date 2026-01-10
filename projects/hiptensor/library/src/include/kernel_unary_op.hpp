/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

// Simple and efficient kernel - single implementation
template <typename DataType, typename UnaryOpType, ck::index_t NPerThread>
__global__ void kernel_hiptensor_unary_op(DataType* __restrict__ data, 
                                                const ck::index_t size,
                                                UnaryOpType unaryOp = UnaryOpType{})
{
    const ck::index_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const ck::index_t globalOffset = gid * NPerThread;

    // Guard against out-of-bounds access
    if(globalOffset >= size)
        return;

    ck::index_t idx = 0;

    // Load data into a local register array
    DataType localData[NPerThread];

    #pragma unroll
    for(ck::index_t i = 0; i < NPerThread; ++i)
    {
        idx = globalOffset + i;
        if(idx < size)
        {
            localData[i] = data[idx];
        }
    }

    // Apply the unary operation
    #pragma unroll
    for(ck::index_t i = 0; i < NPerThread; ++i)
    {
        unaryOp(localData[i], localData[i]);
    }

    // Store the results back to global memory
    #pragma unroll
    for(ck::index_t i = 0; i < NPerThread; ++i)
    {
        idx = globalOffset + i;
        if(idx < size)
        {
            data[idx] = localData[i];  
        }
    }
}