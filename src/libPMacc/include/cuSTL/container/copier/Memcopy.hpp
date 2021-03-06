/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#ifndef CUDAWRAPPERMEMCOPY_HPP
#define CUDAWRAPPERMEMCOPY_HPP

#include "math/vector/Size_t.hpp"
#include <types.h>

namespace PMacc
{
namespace cudaWrapper
{
    
namespace flags
{
struct Memcopy
{
    enum Direction {hostToDevice = 0, deviceToHost, hostToHost, deviceToDevice};
};
}
    
template<int dim>
struct Memcopy;

template<>
struct Memcopy<1>
{
    template<typename Type>
    void operator()(Type* dest, const math::Size_t<0>, 
                    const Type* source, const math::Size_t<0>, const math::Size_t<1>& size,
                    flags::Memcopy::Direction direction)
    {
            const cudaMemcpyKind kind[] = {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                                     cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice};
            CUDA_CHECK_NO_EXCEP(cudaMemcpy(dest, source, sizeof(Type) * size.x(), kind[direction]));
    }                 
};

template<>
struct Memcopy<2u>
{
    template<typename Type>
    void operator()(Type* dest, const math::Size_t<1> pitchDest, 
                    const Type* source, const math::Size_t<1> pitchSource, const math::Size_t<2u>& size,
                    flags::Memcopy::Direction direction)
    {
            const cudaMemcpyKind kind[] = {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                                     cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice};
                                     
            CUDA_CHECK_NO_EXCEP(cudaMemcpy2D(dest, pitchDest.x(), source, pitchSource.x(), sizeof(Type) * size.x(), size.y(),
                         kind[direction]));
    }                    
};

template<>
struct Memcopy<3>
{
    template<typename Type>
    void operator()(Type* dest, const math::Size_t<2u> pitchDest, 
                    Type* source, const math::Size_t<2u> pitchSource, const math::Size_t<3>& size,
                    flags::Memcopy::Direction direction)
    {
            const cudaMemcpyKind kind[] = {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                                     cudaMemcpyHostToHost, cudaMemcpyDeviceToDevice};
                                     
            cudaPitchedPtr pitchedPtrDest;
            pitchedPtrDest.pitch = pitchDest.x(); pitchedPtrDest.ptr = dest;
            pitchedPtrDest.xsize = size.x(); pitchedPtrDest.ysize = size.y();
            cudaPitchedPtr pitchedPtrSource;
            pitchedPtrSource.pitch = pitchSource.x(); pitchedPtrSource.ptr = source;
            pitchedPtrSource.xsize = size.x(); pitchedPtrSource.ysize = size.y();
            
            cudaMemcpy3DParms params;
            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0,0,0);
            params.srcPtr = pitchedPtrSource;
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(0,0,0);
            params.dstPtr = pitchedPtrDest;
            params.extent = make_cudaExtent(size.x() * sizeof(Type), size.y(), size.z());
            params.kind = kind[direction];
            CUDA_CHECK_NO_EXCEP(cudaMemcpy3D(&params));
    }                    
};
    
} // cudaWrapper
} // PMacc

#endif //CUDAWRAPPERMEMCOPY_HPP
