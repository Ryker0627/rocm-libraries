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

#include <ck/utility/math.hpp>
#include <hiptensor/hiptensor_types.h>

namespace hiptensor
{
    // Helper to convert to float for computation
    template<typename T>
    __host__ __device__ inline float toFloat(T const& x) {
        return static_cast<float>(x);
    }
    
    template<>
    __host__ __device__ inline float toFloat<ck::bhalf_t>(ck::bhalf_t const& x) {
        return ck::type_convert<float, ck::bhalf_t>(x);
    }
    
    // Helper to convert from float
    template<typename T>
    __host__ __device__ inline T fromFloat(float const& x) {
        return static_cast<T>(x);
    }
    
    template<>
    __host__ __device__ inline ck::bhalf_t fromFloat<ck::bhalf_t>(float const& x) {
        return ck::type_convert<ck::bhalf_t, float>(x);
    }
    
    // Helper to check if type needs float conversion
    template<typename T>
    struct NeedsFloatConversion {
        static constexpr bool value = 
            std::is_same<T, ck::half_t>::value || 
            std::is_same<T, ck::bhalf_t>::value;
    };
    
    // Specific operation functors - each compiles to minimal code
    
    struct IdentityOp {
        template<typename T>
        __host__ __device__ void operator()(T& y, T const& x) const { y = x; }
    };
    
    struct SqrtOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::sqrt(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::sqrt(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct ReluOp {
        template<typename T>
        __host__ __device__ void operator()(T& y, T const& x) const { 
            y = x > static_cast<T>(0) ? x : static_cast<T>(0); 
        }
    };
    
    struct RcpOp {
        template<typename T>
        __host__ __device__ void operator()(T& y, T const& x) const { 
            y = static_cast<T>(1) / x; 
        }
    };
    
    struct SigmoidOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = static_cast<T>(1) / (static_cast<T>(1) + ck::math::exp(-x)); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = 1.0f / (1.0f + ck::math::exp(-tempX));
            y = fromFloat<T>(tempY);
        }
    };
    
    struct TanhOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::tanh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::tanh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct ExpOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::exp(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::exp(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct LogOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::log(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::log(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AbsOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::abs(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::abs(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct NegOp {
        template<typename T>
        __host__ __device__ void operator()(T& y, T const& x) const { y = -x; }
    };
    
    struct SinOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::sin(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::sin(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct CosOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::cos(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::cos(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct TanOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::tan(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::tan(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct SinhOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::sinh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::sinh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct CoshOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::cosh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::cosh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AsinOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::asin(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::asin(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AcosOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::acos(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::acos(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AtanOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::atan(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::atan(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AsinhOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::asinh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::asinh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AcoshOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::acosh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::acosh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct AtanhOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::atanh(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::atanh(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct CeilOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::ceil(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::ceil(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct FloorOp {
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<!NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            y = ck::math::floor(x); 
        }
        
        template<typename T>
        __host__ __device__ 
        std::enable_if_t<NeedsFloatConversion<T>::value, void>
        operator()(T& y, T const& x) const { 
            float tempX = toFloat(x);
            float tempY = ck::math::floor(tempX);
            y = fromFloat<T>(tempY);
        }
    };
    
    struct ConjOp {
        template<typename T>
        __host__ __device__ void operator()(T& y, T const& x) const { 
            y = x;  // For real types, conjugate is identity
        }
    };
    
} // namespace hiptensor

