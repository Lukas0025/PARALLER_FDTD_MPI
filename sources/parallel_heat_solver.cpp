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

#define X 1
#define Y 0

#define TOTAL_SIZE(A) (A[X] * A[Y])
#define PTR(ARRAY) &(ARRAY[0])

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
    MPI_Scatterv(
        PTR(m_materialProperties.GetDomainParams()),
        PTR(this->localGridScatterCounts),
        PTR(this->localGridDisplacement),
        this->MPILocalGridResized_T,
        PTR(this->localGrid),
        TOTAL_SIZE(this->localGridSizes),
        MPI_FLOAT,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );

    DEBUG_PRINT(MPI_ROOT_RANK, "Scatter done \n");

}

ParallelHeatSolver::~ParallelHeatSolver()
{

}

void ParallelHeatSolver::CreateTypes() {

    int starts[2] = {0, 0}; // inital position for all processes
    MPI_Type_create_subarray(2, PTR(this->globalGridSizes), PTR(this->localGridSizes), starts, MPI_ORDER_C, MPI_FLOAT, &this->MPILocalGrid_T);
    MPI_Type_create_resized(this->MPILocalGrid_T, 0, this->localGridSizes[X] * sizeof(float), &this->MPILocalGridResized_T);
    MPI_Type_commit(&this->MPILocalGrid_T);
    MPI_Type_commit(&this->MPILocalGridResized_T);

    //compute displasement for all subarrays
    this->localGridDisplacement .reserve(TOTAL_SIZE(this->localGridCounts));
    this->localGridScatterCounts.reserve(TOTAL_SIZE(this->localGridCounts));
    for (int i = 0; i < TOTAL_SIZE(this->localGridCounts); i++) {
        this->localGridDisplacement[i]  = i + (i / this->localGridCounts[X]) * (this->localGridSizes[Y] - 1) * this->localGridCounts[X];
        this->localGridScatterCounts[i] = 1;

        DEBUG_PRINT(MPI_ROOT_RANK, "Calculated displacement for %d submeterix is %d \n", i, this->localGridDisplacement[i]);
    }

}

void ParallelHeatSolver::Decompose() {
    //set localgrids count to default
    this->localGridCounts[Y] = 1;
    this->localGridCounts[X] = 1;

    //get localgrids counts
    m_simulationProperties.GetDecompGrid(this->localGridCounts[X], this->localGridCounts[Y]);

    //Calculate real localgrids sizes
    this->localGridSizes[Y] = m_materialProperties.GetEdgeSize() / this->localGridCounts[Y]; 
    this->localGridSizes[X] = m_materialProperties.GetEdgeSize() / this->localGridCounts[X];

    DEBUG_PRINT(MPI_ROOT_RANK, "Calculated decompose is XSize: %d, YSize: %d, XCount: %d, YCount: %d \n",
        this->localGridSizes[Y],
        this->localGridSizes[X],
        this->localGridCounts[Y],
        this->localGridCounts[X]
    );

    //distribute generic info between ranks
    MPI_Bcast(this->localGridSizes,  2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(this->localGridCounts, 2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    //recompute global grid size by local grid info
    this->globalGridSizes[Y] = this->localGridSizes[Y] * this->localGridCounts[Y];
    this->globalGridSizes[X] = this->localGridSizes[X] * this->localGridCounts[X];

    DEBUG_PRINT(MPI_ALL_RANKS, "Global grid size is X: %d Y: %d \n", this->globalGridSizes[Y], this->globalGridSizes[X]);

    //allocate space for localGrid
    this->localGrid.reserve(this->localGridSizes[Y] * this->localGridSizes[X]);

    //crate topology
    int  dims[2]    = {this->localGridCounts[Y], this->localGridCounts[X]};
    int  periods[2] = {false, false}; // no cycle on Y and X
    bool reorder    = true;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &this->MPIGridComm);

    // Get my coordinates in the new communicator
    MPI_Cart_coords(this->MPIGridComm, this->m_rank, 2, this->m_coords);

    DEBUG_PRINT(MPI_ALL_RANKS, "I am located at (%03d, %03d)\n", this->m_coords[Y], this->m_coords[X]);
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

    MPI_Gatherv(
        PTR(this->localGrid),
        this->localGridSizes[Y] * this->localGridSizes[X],
        MPI_FLOAT,
        PTR(outResult),
        PTR(this->localGridScatterCounts),
        PTR(this->localGridDisplacement),
        this->MPILocalGridResized_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );
}
