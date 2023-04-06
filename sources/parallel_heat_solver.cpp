/**
 * @file    parallel_heat_solver.cpp
 * @author  xlogin00 <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-MM-DD
 */

#include "parallel_heat_solver.h"

using namespace std;

//============================================================================//
//                            *** BEGIN: NOTE ***
//
// Implement methods of your ParallelHeatSolver class here.
// Freely modify any existing code in ***THIS FILE*** as only stubs are provided 
// to allow code to compile.
//
//                             *** END: NOTE ***
//============================================================================//

/**
 * C printf routine with selection which rank prints.
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string.
 * @param ...    - other parameters.
 */
void ParallelHeatSolver::mpiPrintf(int who, const char* __restrict__ format, ...) {
    if ((who == MPI_ALL_RANKS) || (who == this->m_rank)) {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
    }
}

ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps,
                                       MaterialProperties &materialProps)
    : BaseHeatSolver (simulationProps, materialProps)
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    
    // Creating EMPTY HDF5 handle using RAII "AutoHandle" type
    //
    // AutoHandle<hid_t> myHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
    //
    // This can be particularly useful when creating handle as class member!
    // Handle CANNOT be assigned using "=" or copy-constructor, yet it can be set
    // using ".Set(/* handle */, /* close/free function */)" as in:
    // myHandle.Set(H5Fopen(...), H5Fclose);
    
    // Requested domain decomposition can be queried by
    // m_simulationProperties.GetDecompGrid(/* TILES IN X */, /* TILES IN Y */)
    this->Decompose();
    this->CreateTypes();

    //distribute init data
    MPI_Scatter(
        &m_materialProperties.GetDomainParams()[0],
        1,
        this->MPIGlobalGrid_T,
        &this->localGrid[0],
        this->localGridSizes[0] * this->localGridSizes[1],
        this->MPILocalGrid_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );

    DEBUG_PRINT(MPI_ROOT_RANK, "Scatter done \n",
    );

}

ParallelHeatSolver::~ParallelHeatSolver()
{

}

void ParallelHeatSolver::CreateTypes() {

    int starts[2] = {this->m_coords[0] * this->localGridSizes[0], this->m_coords[1] * this->localGridSizes[1]}; // inital position for all processes
    MPI_Type_create_subarray(2, &this->globalGridSizes[0], &this->localGridSizes[0], starts, MPI_ORDER_C, MPI_FLOAT, &this->MPILocalGrid_T);
    MPI_Type_commit(&this->MPILocalGrid_T);
    
    starts[0] = 0; starts[1] = 0;
    MPI_Type_create_subarray(2, &this->globalGridSizes[0], &this->globalGridSizes[0], starts, MPI_ORDER_C, MPI_FLOAT, &this->MPIGlobalGrid_T);
    MPI_Type_commit(&this->MPIGlobalGrid_T);

}

void ParallelHeatSolver::Decompose() {
    //set localgrids count to default
    this->localGridCounts[0] = 1;
    this->localGridCounts[1] = 1;

    //get localgrids counts
    m_simulationProperties.GetDecompGrid(this->localGridCounts[0], this->localGridCounts[1]);

    //Calculate real localgrids sizes
    this->localGridSizes[0] = m_materialProperties.GetEdgeSize() / this->localGridCounts[0]; 
    this->localGridSizes[1] = m_materialProperties.GetEdgeSize() / this->localGridCounts[1];

    DEBUG_PRINT(MPI_ROOT_RANK, "Calculated decompose is XSize: %d, YSize: %d, XCount: %d, YCount: %d \n",
        this->localGridSizes[0],
        this->localGridSizes[1],
        this->localGridCounts[0],
        this->localGridCounts[1]
    );

    //distribute generic info between ranks
    MPI_Bcast(this->localGridSizes,  2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(this->localGridCounts, 2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    //recompute global grid size by local grid info
    this->globalGridSizes[0] = this->localGridSizes[0] * this->localGridCounts[0];
    this->globalGridSizes[1] = this->localGridSizes[1] * this->localGridCounts[1];

    DEBUG_PRINT(MPI_ALL_RANKS, "Global grid size is X: %d Y: %d \n", this->globalGridSizes[0], this->globalGridSizes[1]);

    //allocate space for localGrid
    this->localGrid.reserve(this->localGridSizes[0] * this->localGridSizes[1]);

    //crate topology
    int  dims[2]    = {this->localGridCounts[0], this->localGridCounts[1]};
    int  periods[2] = {false, false}; // no cycle on Y and X
    bool reorder    = true;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &this->MPIGridComm);

    // Get my coordinates in the new communicator
    MPI_Cart_coords(this->MPIGridComm, this->m_rank, 2, this->m_coords);

    DEBUG_PRINT(MPI_ALL_RANKS, "I am located at (%03d, %03d)\n", this->m_coords[0], this->m_coords[1]);

    //distribute parts between ranks
    //MPI_Scatter(rand_nums, elements_per_proc, MPI_FLOAT, sub_rand_nums, elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);


}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
    // UpdateTile(...) method can be used to evaluate heat equation over 2D tile
    //                 in parallel (using OpenMP).
    // NOTE: This method might be inefficient when used for small tiles such as 
    //       2xN or Nx2 (these might arise at edges of the tile)
    //       In this case ComputePoint may be called directly in loop.
    
    // ShouldPrintProgress(N) returns true if average temperature should be reported
    // by 0th process at Nth time step (using "PrintProgressReport(...)").
    
    // Finally "PrintFinalReport(...)" should be used to print final elapsed time and
    // average temperature in column.
    
    //test get it back
    MPI_Gather(
        &this->localGrid[0],
        this->localGridSizes[0] * this->localGridSizes[1],
        this->MPILocalGrid_T,
        &outResult[0],
        this->globalGridSizes[0] * this->globalGridSizes[1],
        this->MPIGlobalGrid_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );
}
