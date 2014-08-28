/**
 * Copyright 2013-2014 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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


#include "Environment.hpp"
#include "eventSystem/EventSystem.hpp"

#include "fields/SimulationFieldHelper.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"
#include "particles/memory/buffers/ParticlesBuffer.hpp"


namespace PMacc
{
    
    template< class T_ParticleBox, class Mapping>
    __host__ void wrapper_kernelDeleteParticles(dim3 grid, dim3 block, T_ParticleBox pb, Mapping mapper)
    { 
        __cudaKernel(kernelDeleteParticles<T_ParticleBox>) (grid, block)(pb, mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc>
    void ParticlesBase<T_ParticleDescription, MappingDesc>::deleteGuardParticles(uint32_t exchangeType)
    {

        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        dim3 grid(mapper.getGridDim());

	wrapper_kernelDeleteParticles(grid, TileSize,particlesBuffer->getDeviceParticleBox(), mapper);
/*
        __cudaKernel(kernelDeleteParticles)
                (grid, TileSize)
                (particlesBuffer->getDeviceParticleBox(), mapper);
*/
    }


    template< class FRAME, class BORDER, class Mapping>
    __host__ void wrapper_kernelBashParticles(dim3 grid, dim3 block, ParticlesBox<FRAME, Mapping::Dim> pb,
                                    ExchangePushDataBox<vint_t, BORDER, Mapping::Dim - 1 > border,
                                    Mapping mapper)
    { 
            __cudaKernel(kernelBashParticles<FRAME>)
                    (grid, block)(pb,border, mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc>
    void ParticlesBase<T_ParticleDescription, MappingDesc>::bashParticles(uint32_t exchangeType)
    {
        if (particlesBuffer->hasSendExchange(exchangeType))
        {
            ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

            particlesBuffer->getSendExchangeStack(exchangeType).setCurrentSize(0);
            dim3 grid(mapper.getGridDim());

	    wrapper_kernelBashParticles(grid, TileSize, particlesBuffer->getDeviceParticleBox(),
                    particlesBuffer->getSendExchangeStack(exchangeType).getDeviceExchangePushDataBox(), mapper);
/*
            __cudaKernel(kernelBashParticles)
                    (grid, TileSize)
                    (particlesBuffer->getDeviceParticleBox(),
                    particlesBuffer->getSendExchangeStack(exchangeType).getDeviceExchangePushDataBox(), mapper);
*/
        }
    }

    template<class FRAME, class BORDER, class Mapping>
    __host__ void wrapper_kernelInsertParticles(dim3 grid, dim3 block, ParticlesBox<FRAME, Mapping::Dim> pb,
                                      ExchangePopDataBox<vint_t, BORDER, Mapping::Dim - 1 > border,
                                      Mapping mapper)
    {
                __cudaKernel(kernelInsertParticles<FRAME>)(grid, block)
                        (pb, border, mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc>
    void ParticlesBase<T_ParticleDescription, MappingDesc>::insertParticles(uint32_t exchangeType)
    {
        if (particlesBuffer->hasReceiveExchange(exchangeType))
        {

            dim3 grid(particlesBuffer->getReceiveExchangeStack(exchangeType).getHostCurrentSize());
            if (grid.x != 0)
            {
              //  std::cout<<"insert = "<<grid.x()<<std::endl;
                ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
		wrapper_kernelInsertParticles(grid, TileSize, particlesBuffer->getDeviceParticleBox(),
                        particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                        mapper);
/*
                __cudaKernel(kernelInsertParticles)
                        (grid, TileSize)
                        (particlesBuffer->getDeviceParticleBox(),
                        particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                        mapper);
*/
                }
        }
    }

    template<typename T_ParticleDescription, class MappingDesc>
    EventTask ParticlesBase<T_ParticleDescription, MappingDesc>::asyncCommunication(EventTask event)
    {
        EventTask ret;
        __startTransaction(event);
        Environment<>::get().ParticleFactory().createTaskParticlesReceive(*this);
        ret = __endTransaction();

        __startTransaction(event);
        Environment<>::get().ParticleFactory().createTaskParticlesSend(*this);
        ret += __endTransaction();
        return ret;
    }

} //namespace PMacc


