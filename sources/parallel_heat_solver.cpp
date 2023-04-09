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
#define HALO_SIZE 2

#define UP        0
#define DOWN      1
#define LEFT      2
#define RIGHT     3

#define COOLER_TEMP 1
#define AIR_FLOW    0

#define NONE_TAG 0

#define UP_HALO(A)      &(A[0])
#define DOWN_HALO(A)    &(A[this->downHeloPos])
#define FIRST_LINE(A)   &(A[this->dataStartPos])
#define LAST_2_LINES(A) &(A[this->downHeloPos - this->localGridRowSize * 2])


#define TOTAL_SIZE(A) (A[X] * A[Y])
#define TOTAL_SIZE_WITH_HALO(A) ((A[X] + 2 * HALO_SIZE) * (A[Y] + 2 * HALO_SIZE))
#define GET(X_POS, Y_POS, ARRAY, ROW_SIZE) ARRAY[(Y_POS * ROW_SIZE + X_POS)]
#define LOCAL_GET(X_POS, Y_POS, ARRAY) GET(X_POS, Y_POS, ARRAY, this->localGridSizes[X])
#define LOCAL_HALO_GET(X_POS, Y_POS, ARRAY) GET(X_POS, Y_POS, ARRAY, (this->localGridSizes[X] + 2 * HALO_SIZE))

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

    DEBUG_PRINT(MPI_ROOT_RANK, "Init done \n");

}

ParallelHeatSolver::~ParallelHeatSolver()
{

}

void ParallelHeatSolver::CreateTypes() {

    MPI_Type_create_subarray(2, this->globalGridSizes, this->localGridSizes, (const int[]){0, 0}, MPI_ORDER_C, MPI_FLOAT, &this->MPILocalGridResized_T);
    MPI_Type_create_resized(this->MPILocalGridResized_T, 0, this->localGridSizes[X] * sizeof(float), &this->MPILocalGridResized_T);
    MPI_Type_commit(&this->MPILocalGridResized_T);

    //compute displasement for all subarrays
    this->localGridDisplacement .reserve(TOTAL_SIZE(this->localGridCounts));
    this->localGridScatterCounts.reserve(TOTAL_SIZE(this->localGridCounts));
    for (int i = 0; i < TOTAL_SIZE(this->localGridCounts); i++) {
        this->localGridDisplacement[i]  = i + (i / this->localGridCounts[X]) * (this->localGridSizes[Y] - 1) * this->localGridCounts[X];
        this->localGridScatterCounts[i] = 1;

        DEBUG_PRINT(MPI_ROOT_RANK, "Calculated displacement for %d submeterix is %d \n", i, this->localGridDisplacement[i]);
    }

    const int haloStarts[2] =  {this->neighbours[UP] ? HALO_SIZE : 0, this->neighbours[LEFT] ? HALO_SIZE : 0};

    MPI_Type_create_subarray(2, (const int[]){this->localGridSizes[Y] + 2 * HALO_SIZE, this->localGridSizes[X] + 2 * HALO_SIZE}, this->localGridSizes, haloStarts, MPI_ORDER_C, MPI_FLOAT, &this->MPILocalGridWithHalo_T);
    MPI_Type_commit(&this->MPILocalGridWithHalo_T);

    MPI_Type_create_subarray(2, (const int[]){this->localGridSizes[Y] + 2 * HALO_SIZE, this->localGridSizes[X] + 2 * HALO_SIZE}, this->localGridSizes, haloStarts, MPI_ORDER_C, MPI_INT, &this->MPILocalINTGridWithHalo_T);
    MPI_Type_commit(&this->MPILocalINTGridWithHalo_T);

    this->localGridSizesWithHalo[X] = this->localGridSizes[X] + haloStarts[X] + (this->neighbours[RIGHT] ? HALO_SIZE : 0);
    this->localGridSizesWithHalo[Y] = this->localGridSizes[Y] + haloStarts[Y] + (this->neighbours[DOWN]  ? HALO_SIZE : 0);

    //compute useful indexes
    this->localGridRowSize = this->localGridSizes[X] + 2 * HALO_SIZE;
    this->dataStartPos     = this->localGridRowSize * 2;
    this->downHeloPos      = TOTAL_SIZE(this->localGridSizesWithHalo) - 2 * (this->localGridRowSize * HALO_SIZE);
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

    //crate topology
    int  dims[2]    = {this->localGridCounts[Y], this->localGridCounts[X]};
    int  periods[2] = {false, false}; // no cycle on Y and X
    bool reorder    = true;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &this->MPIGridComm);

    // Get my coordinates in the new communicator
    MPI_Cart_coords(this->MPIGridComm, this->m_rank, 2, this->m_coords);

    //get my neighbours
    this->neighbours[UP]    = this->m_coords[Y] > 0;
    this->neighbours[DOWN]  = this->m_coords[Y] < this->localGridCounts[Y] - 1;
    this->neighbours[LEFT]  = this->m_coords[X] > 0;
    this->neighbours[RIGHT] = this->m_coords[X] < this->localGridCounts[X] - 1;

    //get neighbours ranks
    MPI_Cart_shift(this->MPIGridComm, 1, 1, &(this->neighboursRanks[LEFT]), &(this->neighboursRanks[RIGHT]));
    MPI_Cart_shift(this->MPIGridComm, 0, 1, &(this->neighboursRanks[UP]), &(this->neighboursRanks[DOWN]   ));

    DEBUG_PRINT(MPI_ALL_RANKS, "I am located at (%03d, %03d)\n", this->m_coords[Y], this->m_coords[X]);
}

int ParallelHeatSolver::HaloXCHG(MPI_Request req[8], float* array) {

    int reqCount = 0;

    if (this->neighbours[UP]) {
        MPI_Isend(
            FIRST_LINE(array),
            this->localGridRowSize * 2,
            MPI_FLOAT,
            this->neighboursRanks[UP],
            NONE_TAG, //put self to tag
            this->MPIGridComm,
            &req[reqCount]
        );

        MPI_Irecv(
            UP_HALO(array),
            this->localGridRowSize * 2,
            MPI_FLOAT,
            this->neighboursRanks[UP],
            NONE_TAG,
            this->MPIGridComm,
            &req[reqCount + 1]
        );

        reqCount += 2;
    }
    
    if (this->neighbours[DOWN]) {
        MPI_Isend(
            LAST_2_LINES(array),
            this->localGridRowSize * 2,
            MPI_FLOAT,
            this->neighboursRanks[DOWN],
            NONE_TAG,
            this->MPIGridComm,
            &req[reqCount]
        );

        MPI_Irecv(
            DOWN_HALO(array),
            this->localGridRowSize * 2,
            MPI_FLOAT,
            this->neighboursRanks[DOWN],
            NONE_TAG,
            this->MPIGridComm,
            &req[reqCount + 1]
        );

        reqCount += 2;
    }

    return reqCount;
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

    //reserve space
    this->localTempGrid        .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localTempGrid2       .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localDomainParamsGrid.reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localDomainMapGrid   .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));

    float floatConfig[2] = {m_simulationProperties.GetAirFlowRate(), m_materialProperties.GetCoolerTemp()};
    int   intConfig  [2] = {m_simulationProperties.GetNumIterations()};

    MPI_Bcast(floatConfig, 2, MPI_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(intConfig,   1, MPI_INT,   MPI_ROOT_RANK, MPI_COMM_WORLD);

    //distribute inital data
    MPI_Scatterv(
        m_materialProperties.GetDomainParams().data(),
        this->localGridScatterCounts.data(),
        this->localGridDisplacement.data(),
        this->MPILocalGridResized_T,
        this->localDomainParamsGrid.data(),
        1,
        this->MPILocalGridWithHalo_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );

    MPI_Scatterv(
        m_materialProperties.GetDomainMap().data(),
        this->localGridScatterCounts.data(),
        this->localGridDisplacement.data(),
        this->MPILocalGridResized_T,
        this->localDomainMapGrid.data(),
        1,
        this->MPILocalINTGridWithHalo_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );
    
    MPI_Scatterv(
        m_materialProperties.GetInitTemp().data(),
        this->localGridScatterCounts.data(),
        this->localGridDisplacement.data(),
        this->MPILocalGridResized_T,
        this->localTempGrid.data(),
        1,
        this->MPILocalGridWithHalo_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );

    //send halozones to neighbours
    //if (this->neighbours[LEFT]) 
    //if (this->neighbours[RIGHT])

    MPI_Request req[8];
    auto reqCount = this->HaloXCHG(req, this->localTempGrid.data());
    MPI_Status status[8];
    MPI_Waitall(reqCount, req, status);

    //copy correctly halozones to second array
    for (unsigned i = 0; i < TOTAL_SIZE_WITH_HALO(this->localGridSizes); i++) {
        this->localTempGrid2[i] = this->localTempGrid[i];
    }

    float *workTempArrays[] = { this->localTempGrid.data(), this->localTempGrid2.data()};

      // 3. Begin iterative simulation main loop
    for(size_t iter = 0; iter < intConfig[0]; ++iter) {
        // 4. Compute new temperature for each point in the domain (except borders)
        // border temperatures should remain constant (plus our stencil is +/-2 points).
        for(size_t i = HALO_SIZE; i < this->localGridSizesWithHalo[Y] - HALO_SIZE; ++i) {
            for(size_t j = HALO_SIZE; j < this->localGridSizesWithHalo[X] - HALO_SIZE; ++j) {
                ComputePoint(
                    workTempArrays[0], workTempArrays[1],
                    this->localDomainParamsGrid.data(),
                    this->localDomainMapGrid.data(),
                    i, j,
                    this->localGridSizes[X] + 2 * HALO_SIZE,
                    floatConfig[AIR_FLOW],
                    floatConfig[COOLER_TEMP]
                );
            }
        }

        MPI_Request req[8];
        auto reqCount = this->HaloXCHG(req, workTempArrays[1]);
    MPI_Status status[8];
    MPI_Waitall(reqCount, req, status);
        std::swap(workTempArrays[0], workTempArrays[1]);
    }

    MPI_Gatherv(
        this->localTempGrid.data(),
        1,
        this->MPILocalGridWithHalo_T,
        outResult.data(),
        this->localGridScatterCounts.data(),
        this->localGridDisplacement.data(),
        this->MPILocalGridResized_T,
        MPI_ROOT_RANK,
        this->MPIGridComm
    );
}
