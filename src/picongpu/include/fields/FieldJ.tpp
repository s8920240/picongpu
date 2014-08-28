/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
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

#include <iostream>
#include "simulation_defines.hpp"
#include "FieldJ.hpp"
#include "fields/FieldJ.kernel"


#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"

#include "mappings/kernel/AreaMapping.hpp"
#include "mappings/kernel/StrideMapping.hpp"
#include "mappings/kernel/ExchangeMapping.hpp"
#include "fields/tasks/FieldFactory.hpp"

#include "fields/numericalCellTypes/NumericalCellTypes.hpp"

#include "math/vector/compile-time/Vector.hpp"

namespace picongpu
{

using namespace PMacc;

FieldJ::FieldJ( MappingDesc cellDescription ) :
SimulationFieldHelper<MappingDesc>( cellDescription ),
fieldJ( cellDescription.getGridLayout( ) ), fieldE( NULL )
{
    typedef currentSolver::CurrentSolver ParticleCurrentSolver;

    const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout( ).getDataSpaceWithoutGuarding( );

    typedef typename GetMargin<ParticleCurrentSolver>::LowerMargin LowerMargin;
    typedef typename GetMargin<ParticleCurrentSolver>::UpperMargin UpperMargin;

    const DataSpace<simDim> originGuard( LowerMargin( ).toRT( ) );
    const DataSpace<simDim> endGuard( UpperMargin( ).toRT( ) );

    /*go over all directions*/
    for ( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
    {
        DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim > ( i );
        /*guarding cells depend on direction
         */
        DataSpace<simDim> guardingCells;
        for ( uint32_t d = 0; d < simDim; ++d )
        {
            /*originGuard and endGuard are switch because we send data
             * e.g. from left I get endGuardingCells and from right I originGuardingCells
             */
            switch ( relativMask[d] )
            {
                /*receive from negativ side positiv (end) garding cells*/
            case -1: guardingCells[d] = endGuard[d];
                break;
                /*receive from positiv side negativ (origin) garding cells*/
            case 1: guardingCells[d] = originGuard[d];
                break;
            case 0: guardingCells[d] = coreBorderSize[d];
                break;
            };

        }
        // std::cout << "ex " << i << " x=" << guardingCells[0] << " y=" << guardingCells[1] << " z=" << guardingCells[2] << std::endl;
        fieldJ.addExchangeBuffer( i, guardingCells, FIELD_J );
    }
}

FieldJ::~FieldJ( )
{
}

SimulationDataId FieldJ::getUniqueId()
{
    return getName();
}

void FieldJ::synchronize( )
{
    fieldJ.deviceToHost( );
}

GridBuffer<FieldJ::ValueType, simDim> &FieldJ::getGridBuffer( )
{
    return fieldJ;
}

EventTask FieldJ::asyncCommunication( EventTask serialEvent )
{
    EventTask ret;
    __startTransaction( serialEvent );
    FieldFactory::getInstance( ).createTaskFieldReceiveAndInsert( *this );
    ret = __endTransaction( );

    __startTransaction( serialEvent );
    FieldFactory::getInstance( ).createTaskFieldSend( *this );
    ret += __endTransaction( );
    return ret;
}

template<class Mapping>
__host__ void wrapper_kernelBashCurrent(dim3 grid, dim3 block, J_DataBox fieldJ,
                                  J_DataBox targetJ,
                                  DataSpace<simDim> exchangeSize,
                                  DataSpace<simDim> direction,
                                  Mapping mapper)
{
	__cudaKernel( kernelBashCurrent<Mapping> )
        ( grid, block )
        ( fieldJ, targetJ,exchangeSize, direction,  mapper );
}

void FieldJ::bashField( uint32_t exchangeType )
{
    ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

    dim3 grid = mapper.getGridDim( );

    const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
    wrapper_kernelBashCurrent(grid, mapper.getSuperCellSize( ), fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction,
          mapper);
/*    __cudaKernel( kernelBashCurrent )
        ( grid, mapper.getSuperCellSize( ) )
        ( fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getSendExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction,
          mapper );
*/
}

template<class Mapping>
__host__ void wrapper_kernelInsertCurrent(dim3 grid, dim3 block, J_DataBox fieldJ,
                                    J_DataBox sourceJ,
                                    DataSpace<simDim> exchangeSize,
                                    DataSpace<simDim> direction,
                                    Mapping mapper)
{
    __cudaKernel( kernelInsertCurrent<Mapping> ) (grid, block) ( fieldJ, sourceJ, exchangeSize, direction, mapper );
}

void FieldJ::insertField( uint32_t exchangeType )
{
    ExchangeMapping<GUARD, MappingDesc> mapper( this->cellDescription, exchangeType );

    dim3 grid = mapper.getGridDim( );

    const DataSpace<simDim> direction = Mask::getRelativeDirections<simDim > ( mapper.getExchangeType( ) );
    wrapper_kernelInsertCurrent(grid, mapper.getSuperCellSize( ), fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction, mapper );
/*
    __cudaKernel( kernelInsertCurrent )
        ( grid, mapper.getSuperCellSize( ) )
        ( fieldJ.getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataBox( ),
          fieldJ.getReceiveExchange( exchangeType ).getDeviceBuffer( ).getDataSpace( ),
          direction, mapper );
*/
}

void FieldJ::init( FieldE &fieldE )
{
    this->fieldE = &fieldE;

    Environment<>::get().DataConnector().registerData( *this );
}

GridLayout<simDim> FieldJ::getGridLayout( )
{
    return cellDescription.getGridLayout( );
}

void FieldJ::reset( uint32_t )
{
}

void FieldJ::clear( )
{
    ValueType tmp = float3_X( 0., 0., 0. );
    fieldJ.getDeviceBuffer( ).setValue( tmp );
    //fieldJ.reset(false);
}

typename FieldJ::UnitValueType
FieldJ::getUnit( )
{
    const UnitValueType unitaryVector( 1.0, 1.0, 1.0 );
    return unitaryVector * UNIT_CHARGE / UNIT_TIME / (UNIT_LENGTH * UNIT_LENGTH);
}

std::string
FieldJ::getName( )
{
    return "FieldJ";
}

uint32_t
FieldJ::getCommTag( )
{
    return FIELD_J;
}

template<int workerMultiplier, class BlockDescription_, uint32_t AREA, class JBox, class ParBox, class Mapping, class FrameSolver>
__host__ void wrapper_kernelComputeCurrent(dim3 grid, dim3 block, JBox fieldJ,
                                     ParBox boxPar, FrameSolver frameSolver, Mapping mapper)
{
__cudaKernel( ( kernelComputeCurrent<workerMultiplier,BlockDescription_, AREA, JBox, ParBox, Mapping, FrameSolver> ) )
            ( grid, block)
            ( fieldJ,
              boxPar, frameSolver, mapper );

}

template<uint32_t AREA, class ParticlesClass>
void FieldJ::computeCurrent( ParticlesClass &parClass, uint32_t ) throw (std::invalid_argument )
{
    /** tune paramter to use more threads than cells in a supercell
     *  valid domain: 1 <= workerMultiplier
     */
    const int workerMultiplier =2;

    typedef currentSolver::CurrentSolver ParticleCurrentSolver;
    typedef ComputeCurrentPerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize> FrameSolver;

    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        GetMargin<currentSolver::CurrentSolver>::LowerMargin,
        GetMargin<currentSolver::CurrentSolver>::UpperMargin
        > BlockArea;

    StrideMapping<AREA, simDim, MappingDesc> mapper( cellDescription );
    typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox( );
    FieldJ::DataBoxType jBox = this->fieldJ.getDeviceBuffer( ).getDataBox( );
    FrameSolver solver( DELTA_T );

    DataSpace<simDim> blockSize(mapper.getSuperCellSize( ));
    blockSize[simDim-1]*=workerMultiplier;

    __startAtomicTransaction( __getTransactionEvent( ) );
    do
    {
	wrapper_kernelComputeCurrent<workerMultiplier,BlockArea, AREA>( mapper.getGridDim( ), blockSize, jBox, pBox, solver, mapper);
/*
__cudaKernel( ( kernelComputeCurrent<workerMultiplier,BlockArea, AREA> ) )
            ( mapper.getGridDim( ), blockSize )
            ( jBox,
              pBox, solver, mapper );
*/
    }
    while ( mapper.next( ) );
    __setTransactionEvent( __endTransaction( ) );
}

template<class Mapping>
__host__ void wrapper_kernelAddCurrentToE(dim3 grid, dim3 block, typename FieldE::DataBoxType fieldE,
                                    J_DataBox fieldJ,
                                    Mapping mapper)
{
	__cudaKernel((kernelAddCurrentToE))(grid,block)(fieldE, fieldJ,mapper);
}

template<uint32_t AREA>
void FieldJ::addCurrentToE( )
{

AreaMapping<AREA,MappingDesc> mapper(cellDescription);
wrapper_kernelAddCurrentToE(mapper.getGridDim(),MappingDesc::SuperCellSize::toRT( ).toDim3(),  this->fieldE->getDeviceDataBox( ),
          this->fieldJ.getDeviceBuffer( ).getDataBox( ), mapper );

/*
    __picKernelArea( ( kernelAddCurrentToE ),
                     cellDescription,
                     AREA )
        ( MappingDesc::SuperCellSize::toRT( ).toDim3() )
        ( this->fieldE->getDeviceDataBox( ),
          this->fieldJ.getDeviceBuffer( ).getDataBox( ) );
*/
}

}
