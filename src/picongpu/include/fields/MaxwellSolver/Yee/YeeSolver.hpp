/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */



#pragma once

#include "YeeSolver.def"

#include "types.h"
#include "simulation_defines.hpp"
#include "basicOperations.hpp"

#include "fields/SimulationFieldHelper.hpp"
#include "dataManagement/ISimulationData.hpp"


#include "simulation_classTypes.hpp"
#include "memory/boxes/SharedBox.hpp"
#include "nvidia/functors/Assign.hpp"
#include "mappings/threads/ThreadCollective.hpp"
#include "memory/boxes/CachedBox.hpp"
#include "dimensions/DataSpace.hpp"
#include <fields/FieldE.hpp>
#include <fields/FieldB.hpp>

#include "fields/FieldManipulator.hpp"
#include "fields/LaserPhysics.hpp"

#include <fields/MaxwellSolver/Yee/YeeSolver.kernel>

namespace picongpu
{
namespace yeeSolver
{
using namespace PMacc;


template<class BlockDescription_, class CurlType_, class EBox, class BBox, class Mapping>
__host__ void wrapper_kernelUpdateE(dim3 grid, dim3 block, EBox fieldE, BBox fieldB, Mapping mapper)
{
//        __picKernelArea((kernelUpdateE<BlockDescription_, CurlType_>), cellDescription, AREA)
//                (grid)(fieldE, fieldB);

        __cudaKernel((kernelUpdateE<BlockDescription_, CurlType_>))(grid, block)
                        (fieldE, fieldB,mapper);

}


template<class BlockDescription_, class CurlType_, class EBox, class BBox, class Mapping>
__host__ void wrapper_kernelUpdateBHalf(dim3 grid, dim3 block, BBox fieldB,
                                  EBox fieldE,
                                  Mapping mapper)
{
        __cudaKernel((kernelUpdateBHalf<BlockDescription_, CurlType_>))(grid, block)
                        (fieldB, fieldE,mapper);
/*
        __picKernelArea((kernelUpdateBHalf<BlockArea, CurlE>), cellDescription, AREA)
                (SuperCellSize::toRT().toDim3())
                (this->fieldB->getDeviceDataBox(),
                this->fieldE->getDeviceDataBox());
*/
}


template<class CurlE, class CurlB>
class YeeSolver
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;

    FieldE* fieldE;
    FieldB* fieldB;
    MappingDesc cellDescription;

    template<uint32_t AREA>
    void updateE()
    {
        typedef SuperCellDescription<
                SuperCellSize,
                typename CurlB::LowerMargin,
                typename CurlB::UpperMargin
                > BlockArea;

	AreaMapping<AREA,MappingDesc> mapper(cellDescription);
//	__cudaKernel((kernelUpdateE<BlockArea, CurlB>))(mapper.getGridDim(),SuperCellSize::toRT().toDim3())
//			(this->fieldE->getDeviceDataBox(), this->fieldB->getDeviceDataBox(),mapper);

	wrapper_kernelUpdateE<BlockArea, CurlB>(mapper.getGridDim(),SuperCellSize::toRT().toDim3(), this->fieldE->getDeviceDataBox(), this->fieldB->getDeviceDataBox(),mapper);
/*
        __picKernelArea((kernelUpdateE<BlockArea, CurlB>), cellDescription, AREA)
                (SuperCellSize::toRT().toDim3())
                (this->fieldE->getDeviceDataBox(), this->fieldB->getDeviceDataBox());
*/
    }

    template<uint32_t AREA>
    void updateBHalf()
    {
        typedef SuperCellDescription<
                SuperCellSize,
                typename CurlE::LowerMargin,
                typename CurlE::UpperMargin
                > BlockArea;

	AreaMapping<AREA,MappingDesc> mapper(cellDescription);
//	__cudaKernel((kernelUpdateBHalf<BlockArea, CurlE>))(mapper.getGridDim(),SuperCellSize::toRT().toDim3())(this->fieldB->getDeviceDataBox(),
//                this->fieldE->getDeviceDataBox(),mapper);

        wrapper_kernelUpdateBHalf<BlockArea, CurlE>(mapper.getGridDim(),SuperCellSize::toRT().toDim3(), this->fieldB->getDeviceDataBox(), this->fieldE->getDeviceDataBox(),mapper);
/*
        __picKernelArea((kernelUpdateBHalf<BlockArea, CurlE>), cellDescription, AREA)
                (SuperCellSize::toRT().toDim3())
                (this->fieldB->getDeviceDataBox(),
                this->fieldE->getDeviceDataBox());
*/
    }

public:

    YeeSolver(MappingDesc cellDescription) : cellDescription(cellDescription)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        this->fieldE = &dc.getData<FieldE > (FieldE::getName(), true);
        this->fieldB = &dc.getData<FieldB > (FieldB::getName(), true);
    }

    void update_beforeCurrent(uint32_t)
    {
        updateBHalf < CORE+BORDER >();
        EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());

        updateE<CORE>();
        __setTransactionEvent(eRfieldB);
        updateE<BORDER>();
    }

    void update_afterCurrent(uint32_t currentStep)
    {
        FieldManipulator::absorbBorder(currentStep,this->cellDescription, this->fieldE->getDeviceDataBox());
        if (laserProfile::INIT_TIME > float_X(0.0))
            fieldE->laserManipulation(currentStep);

        EventTask eRfieldE = fieldE->asyncCommunication(__getTransactionEvent());

        updateBHalf < CORE> ();
        __setTransactionEvent(eRfieldE);
        updateBHalf < BORDER > ();

        FieldManipulator::absorbBorder(currentStep,this->cellDescription, fieldB->getDeviceDataBox());

        EventTask eRfieldB = fieldB->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(eRfieldB);
    }
};

} // yeeSolver

} // picongpu
