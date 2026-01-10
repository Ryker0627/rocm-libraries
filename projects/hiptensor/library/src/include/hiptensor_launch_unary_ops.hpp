/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *******************************************************************************/

#pragma once

#include <hiptensor/hiptensor_types.h>
#include "hiptensor_unary_op_types.hpp"

namespace hiptensor
{
    
    // Map operation enum to functor type
    template<hiptensorOperator_t Op>
    struct OpToFunctor;
    
    // Register operations with their corresponding functors
    #define REGISTER_UNARY_OP(OP_ENUM, FUNCTOR) \
        template<> struct OpToFunctor<OP_ENUM> { using type = FUNCTOR; };
    
    // Register all available operations
    REGISTER_UNARY_OP(HIPTENSOR_OP_IDENTITY, IdentityOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_SQRT, SqrtOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_RELU, ReluOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_CONJ, ConjOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_RCP, RcpOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_SIGMOID, SigmoidOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_TANH, TanhOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_EXP, ExpOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_LOG, LogOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ABS, AbsOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_NEG, NegOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_SIN, SinOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_COS, CosOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_TAN, TanOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_SINH, SinhOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_COSH, CoshOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ASIN, AsinOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ACOS, AcosOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ATAN, AtanOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ASINH, AsinhOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ACOSH, AcoshOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_ATANH, AtanhOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_CEIL, CeilOp)
    REGISTER_UNARY_OP(HIPTENSOR_OP_FLOOR, FloorOp)

    // Launch kernel for specific operation
    template<typename DataType, hiptensorOperator_t Op>
    bool launchIfMatch(hiptensorOperator_t unaryOp, void* tensor, std::size_t dataSize, 
                      dim3 grid, dim3 block)
    {
        if(Op == unaryOp)
        {
            using FunctorType = typename OpToFunctor<Op>::type;
            kernel_hiptensor_unary_op<DataType, FunctorType, 1><<<grid, block>>>(
                static_cast<DataType*>(tensor), dataSize, FunctorType{});
            return true;
        }
        return false;
    }
   
    // Define which operations to compile 
    using EnabledUnaryOps = std::integer_sequence<
        hiptensorOperator_t,
        HIPTENSOR_OP_IDENTITY,
        HIPTENSOR_OP_SQRT,
        HIPTENSOR_OP_RELU,
        HIPTENSOR_OP_CONJ,
        HIPTENSOR_OP_RCP,
        HIPTENSOR_OP_SIGMOID,
        HIPTENSOR_OP_TANH,
        HIPTENSOR_OP_EXP,
        HIPTENSOR_OP_LOG,
        HIPTENSOR_OP_ABS,
        HIPTENSOR_OP_NEG,
        HIPTENSOR_OP_SIN,
        HIPTENSOR_OP_COS,
        HIPTENSOR_OP_TAN,
        HIPTENSOR_OP_SINH,
        HIPTENSOR_OP_COSH,
        HIPTENSOR_OP_ASIN,
        HIPTENSOR_OP_ACOS,
        HIPTENSOR_OP_ATAN,
        HIPTENSOR_OP_ASINH,
        HIPTENSOR_OP_ACOSH,
        HIPTENSOR_OP_ATANH,
        HIPTENSOR_OP_CEIL,
        HIPTENSOR_OP_FLOOR
    >;
           
    // Try each operation in the list recursively until a match is found or the list is exhausted
    template<typename DataType, typename OpSeq>
    struct UnaryOpMatch;
    
    // Base case: empty list
    template<typename DataType>
    struct UnaryOpMatch<DataType, std::integer_sequence<hiptensorOperator_t>>
    {
        static bool match(hiptensorOperator_t op, void* tensor, std::size_t dataSize,
                           dim3 grid, dim3 block)
        {
            return false; // Operation not found
        }
    };
    
    // Recursive case: try first operation, then rest
    template<typename DataType, hiptensorOperator_t First, hiptensorOperator_t... Rest>
    struct UnaryOpMatch<DataType, std::integer_sequence<hiptensorOperator_t, First, Rest...>>
    {
        static bool match(hiptensorOperator_t op, void* tensor, std::size_t dataSize,
                           dim3 grid, dim3 block)
        {
            // Try current operation
            if(launchIfMatch<DataType, First>(op, tensor, dataSize, grid, block))
                return true;
            
            // Recursively try remaining operations (compile-time recursion)
            return UnaryOpMatch<DataType, std::integer_sequence<hiptensorOperator_t, Rest...>>
                ::match(op, tensor, dataSize, grid, block);
        }
    };
    
   
    template<typename DataType>
    bool launchUnaryKernelRecurMatch(void* tensor, std::size_t dataSize, hiptensorOperator_t op,
                                   dim3 grid, dim3 block)
    {
        // Compile instantiations for operations in EnabledUnaryOps
        return UnaryOpMatch<DataType, EnabledUnaryOps>::match(
            op, tensor, dataSize, grid, block);
    }
    
} // namespace hiptensor

