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

#include "data_types.hpp"
#include "kernel_unary_op.hpp"
#include "hiptensor_element_wise_operation.hpp"
/*******************************************************************************
 *
 * Unary Operation Tensor Dispatcher
 * 
 * This file includes the dispatch logic for unary operations.
 * The actual implementations are in:
 * - util_host_dispatch.cpp (traditional switch-case)
 * - util_metaprog.cpp (meta-programming selective compilation)
 * 
 * To change which method is used, edit unary_op_dispatch.hpp
 *
 *******************************************************************************/

 #include <iostream>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include "kernel_unary_op.hpp"
#include "hiptensor_unary_op_types.hpp"
#include "hiptensor_launch_unary_ops.hpp"

namespace hiptensor
{
    float unaryOpTensor(const hiptensorDataType_t dataType, void* tensor, 
                            std::size_t dataSize, hiptensorOperator_t op)
    {
        dim3 block(256);
        dim3 grid((dataSize + block.x - 1) / block.x);

        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
        CHECK_HIP_ERROR(hipEventRecord(startEvent));

        bool launched = false;
        switch(dataType)
        {
        case HIPTENSOR_R_16F:
            launched = launchUnaryKernelRecurMatch<ck::half_t>(tensor, dataSize, op, grid, block);
            break;
        case HIPTENSOR_R_16BF:
            launched = launchUnaryKernelRecurMatch<ck::bhalf_t>(tensor, dataSize, op, grid, block);
            break;
        case HIPTENSOR_R_32F:
            launched = launchUnaryKernelRecurMatch<float>(tensor, dataSize, op, grid, block);
            break;
        case HIPTENSOR_R_64F:
            launched = launchUnaryKernelRecurMatch<double>(tensor, dataSize, op, grid, block);
            break;
        default:
            break;
        }

        if(!launched)
        {
            std::cerr << "ERROR: Unary operation " << op << " not compiled."<< std::endl;
        }

        CHECK_HIP_ERROR(hipEventRecord(stopEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

        float elapsedTimeMs = 0.0f;
        CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
        CHECK_HIP_ERROR(hipEventDestroy(startEvent));
        CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

        return elapsedTimeMs;
    }

}
