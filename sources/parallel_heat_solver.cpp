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
#define LEFT_HALO(A)    &(A[0])
#define RIGHT_HALO(A)   &(A[this->rightHeloPos])

#define FIRST_LINE(A)   &(A[this->dataStartPos])
#define LAST_2_LINES(A) &(A[this->downHeloPos - this->localGridRowSize * 2])
#define FIRST_COL(A)    &(A[this->dataStartColPos])
#define LAST_2_COLS(A)  &(A[this->rightHeloPos - 2])

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
    : BaseHeatSolver (simulationProps, materialProps),
    m_fileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
{
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    
    this->Decompose();
    this->CreateTypes();
    this->ReserveFile();

    DEBUG_PRINT(MPI_ROOT_RANK, "Init done \n");

}

ParallelHeatSolver::~ParallelHeatSolver()
{

}

void ParallelHeatSolver::ReserveFile() {
    
    this->FileNameLen = this->m_simulationProperties.GetOutputFileName("par").size();

    MPI_Bcast(&this->FileNameLen, 1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    if (this->FileNameLen <= 0) return; //no file output

    this->FileName.reserve(this->FileNameLen);
    this->FileName = this->m_simulationProperties.GetOutputFileName("par");
    
    MPI_Bcast(this->FileName.data(), this->FileNameLen, MPI_CHAR, MPI_ROOT_RANK, MPI_COMM_WORLD);

    this->UseParallelIO = this->m_simulationProperties.IsUseParallelIO();

    MPI_Bcast(&this->UseParallelIO, 1, MPI_CXX_BOOL, MPI_ROOT_RANK, MPI_COMM_WORLD);

    if (this->UseParallelIO) {
        DEBUG_PRINT(MPI_ALL_RANKS, "Setuping paraller IO to file name is %s\n", this->FileName.c_str());
        m_fileHandle.Set(H5Fcreate(this->FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    } else if (this->m_rank == MPI_ROOT_RANK) {
        DEBUG_PRINT(MPI_ALL_RANKS, "Setuping seq IO to file %s\n", this->FileName.c_str());
        m_fileHandle.Set(H5Fcreate(this->FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    }
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

    DEBUG_PRINT(MPI_ALL_RANKS, "My submetrix start at (%03d, %03d)\n", haloStarts[Y], haloStarts[X]);

    this->localGridSizesWithHalo[X] = this->localGridSizes[X] + haloStarts[X] + (this->neighbours[RIGHT] ? HALO_SIZE : 0);
    this->localGridSizesWithHalo[Y] = this->localGridSizes[Y] + haloStarts[Y] + (this->neighbours[DOWN]  ? HALO_SIZE : 0);

    int  globalMidCol      = this->globalGridSizes[X] / 2;

    //compute useful indexes
    this->middleCol         = globalMidCol - this->localGridDisplacement[this->m_rank % this->localGridCounts[X]] * this->localGridSizes[X];
    this->middleCol         = (this->middleCol >= 0 && this->middleCol < this->localGridSizes[X]) ? this->middleCol + (this->neighbours[LEFT] ? HALO_SIZE : 0): -1;
    this->middleColRootRank = this->localGridCounts[X] / 2;
    this->localGridRowSize  = this->localGridSizes[X] + 2 * HALO_SIZE;
    this->localGridColSize  = this->localGridSizes[Y] + 2 * HALO_SIZE;
    this->dataStartPos      = this->localGridRowSize * 2;
    this->dataStartColPos   = haloStarts[X];
    this->downHeloPos       = TOTAL_SIZE_WITH_HALO(this->localGridSizes) - (this->localGridRowSize * (HALO_SIZE + (!this->neighbours[UP]  ? HALO_SIZE : 0)));
    this->rightHeloPos      = haloStarts[X] + this->localGridSizes[X];

    MPI_Type_contiguous(this->localGridRowSize, MPI_FLOAT, &this->MPILocalGridRow_T);
    MPI_Type_commit(&this->MPILocalGridRow_T);

    MPI_Type_contiguous(this->localGridRowSize, MPI_INT, &this->MPILocalINTGridRow_T);
    MPI_Type_commit(&this->MPILocalINTGridRow_T);

    MPI_Type_vector(this->localGridColSize, 1, this->localGridRowSize, MPI_FLOAT, &this->MPILocalGridCol_T);
    MPI_Type_create_resized(this->MPILocalGridCol_T, 0, 1 * sizeof(float), &this->MPILocalGridCol_T);
    MPI_Type_commit(&this->MPILocalGridCol_T);

    MPI_Type_vector(this->localGridColSize, 1, this->localGridRowSize, MPI_INT, &this->MPILocalINTGridCol_T);
    MPI_Type_create_resized(this->MPILocalINTGridCol_T, 0, 1 * sizeof(int), &this->MPILocalINTGridCol_T);
    MPI_Type_commit(&this->MPILocalINTGridCol_T);
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
        this->localGridSizes[X],
        this->localGridSizes[Y],
        this->localGridCounts[X],
        this->localGridCounts[Y]
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

    //create column comunicator
    //Split the communicator based on the color based on col index
    MPI_Comm_split(this->MPIGridComm, this->m_coords[X], this->m_rank, &this->MPIColComm);
}

void ParallelHeatSolver::HaloMaterialXCHG() {
    int reqCount;
    MPI_Request req[16];
    MPI_Status status[16];

    reqCount = this->HaloXCHG(req, this->localDomainParamsGrid.data());
    MPI_Waitall(reqCount, req, status);

    reqCount = this->HaloINTXCHG(req, this->localDomainMapGrid.data());
    MPI_Waitall(reqCount, req, status);
}

int ParallelHeatSolver::HaloINTXCHG(MPI_Request* req, int* array) {
    int reqCount = 0;

    if (this->neighbours[UP]) {
        MPI_Isend(FIRST_LINE(array), 2, this->MPILocalINTGridRow_T, this->neighboursRanks[UP], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(UP_HALO(array),    2, this->MPILocalINTGridRow_T, this->neighboursRanks[UP], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }
    
    if (this->neighbours[DOWN]) {
        MPI_Isend(LAST_2_LINES(array), 2, this->MPILocalINTGridRow_T, this->neighboursRanks[DOWN], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(DOWN_HALO(array),    2, this->MPILocalINTGridRow_T, this->neighboursRanks[DOWN], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    if (this->neighbours[LEFT]) {
        MPI_Isend(FIRST_COL(array), 2, this->MPILocalINTGridCol_T, this->neighboursRanks[LEFT], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(LEFT_HALO(array), 2, this->MPILocalINTGridCol_T, this->neighboursRanks[LEFT], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    if (this->neighbours[RIGHT]) {
        MPI_Isend(LAST_2_COLS(array), 2, this->MPILocalINTGridCol_T, this->neighboursRanks[RIGHT], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(RIGHT_HALO(array),  2, this->MPILocalINTGridCol_T, this->neighboursRanks[RIGHT], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    return reqCount;
}

int ParallelHeatSolver::HaloXCHG(MPI_Request*    req, float* array) {

    int reqCount = 0;

    if (this->neighbours[UP]) {
        MPI_Isend(FIRST_LINE(array), 2, this->MPILocalGridRow_T, this->neighboursRanks[UP], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(UP_HALO(array),    2, this->MPILocalGridRow_T, this->neighboursRanks[UP], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }
    
    if (this->neighbours[DOWN]) {
        MPI_Isend(LAST_2_LINES(array), 2, this->MPILocalGridRow_T, this->neighboursRanks[DOWN], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(DOWN_HALO(array),    2, this->MPILocalGridRow_T, this->neighboursRanks[DOWN], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    if (this->neighbours[LEFT]) {
        MPI_Isend(FIRST_COL(array), 2, this->MPILocalGridCol_T, this->neighboursRanks[LEFT], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(LEFT_HALO(array), 2, this->MPILocalGridCol_T, this->neighboursRanks[LEFT], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    if (this->neighbours[RIGHT]) {
        MPI_Isend(LAST_2_COLS(array), 2, this->MPILocalGridCol_T, this->neighboursRanks[RIGHT], NONE_TAG, this->MPIGridComm, &req[reqCount]);
        MPI_Irecv(RIGHT_HALO(array),  2, this->MPILocalGridCol_T, this->neighboursRanks[RIGHT], NONE_TAG, this->MPIGridComm, &req[reqCount + 1]);

        reqCount += 2;
    }

    return reqCount;
}

float ParallelHeatSolver::ComputeColSum(const float *data, int index) {

    float tempSum = 0.0f;

    if (index < 0) return tempSum;

    for(size_t i = (this->neighbours[UP] ? HALO_SIZE : 0); i < this->localGridSizesWithHalo[Y] - (this->neighbours[DOWN] ? HALO_SIZE : 0); ++i) {
        tempSum += data[i * this->localGridRowSize + index];
    }

    return tempSum;
}

void ParallelHeatSolver::SaveToFile(const float *data, size_t iter) {
    if (this->UseParallelIO) {

    } else {
        //gather the data to ROOT first
        std::vector<float, AlignedAllocator<float>> outResult;
        outResult.reserve(TOTAL_SIZE(this->globalGridSizes));
        
        MPI_Gatherv(
            data,
            1,
            this->MPILocalGridWithHalo_T,
            outResult.data(),
            this->localGridScatterCounts.data(),
            this->localGridDisplacement.data(),
            this->MPILocalGridResized_T,
            MPI_ROOT_RANK,
            this->MPIGridComm
        );

        if (this->m_rank == MPI_ROOT_RANK) {
            StoreDataIntoFile(this->m_fileHandle, iter, outResult.data());
        }
    }
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
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

    //exchage halozones material info
    this->HaloMaterialXCHG();

    MPI_Request reqXCHG[16];
    int         reqCount = 0;
    
    reqCount = this->HaloXCHG(reqXCHG, this->localTempGrid.data());

    MPI_Waitall(reqCount, reqXCHG, MPI_STATUS_IGNORE);

    

    //copy correctly halozones to second array
    for (unsigned i = 0; i < TOTAL_SIZE_WITH_HALO(this->localGridSizes); i++) {
        this->localTempGrid2[i] = this->localTempGrid[i];
    }    

    float *workTempArrays[] = { this->localTempGrid.data(), this->localTempGrid2.data()};

    //Note the start time
    double startTime = MPI_Wtime();
    float  middleColAvgTemp = 0;

    // Begin iterative simulation main loop
    for(size_t iter = 0; iter < intConfig[0]; ++iter) {

        //Loop over halo XCHG zones
        //frist 2 lines
        for (size_t i = HALO_SIZE;  i < HALO_SIZE * 2; ++i)  {
            for (size_t j = HALO_SIZE; j < this->localGridSizesWithHalo[X] - HALO_SIZE; ++j) {
                ComputePoint(
                    workTempArrays[0], workTempArrays[1],
                    this->localDomainParamsGrid.data(),
                    this->localDomainMapGrid.data(),
                    i, j,
                    this->localGridRowSize,
                    floatConfig[AIR_FLOW],
                    floatConfig[COOLER_TEMP]
                );
            }
        }

        //last 2 lines
        for (size_t i = this->localGridSizesWithHalo[Y] - HALO_SIZE * 2;  i < this->localGridSizesWithHalo[Y] - HALO_SIZE; ++i)  {
            for (size_t j = HALO_SIZE; j < this->localGridSizesWithHalo[X] - HALO_SIZE; ++j) {
                ComputePoint(
                    workTempArrays[0], workTempArrays[1],
                    this->localDomainParamsGrid.data(),
                    this->localDomainMapGrid.data(),
                    i, j,
                    this->localGridRowSize,
                    floatConfig[AIR_FLOW],
                    floatConfig[COOLER_TEMP]
                );
            }
        }

        //left and right halos
        for(size_t i = HALO_SIZE * 2; i < this->localGridSizesWithHalo[Y] - HALO_SIZE * 2; ++i) {
            for (size_t j = HALO_SIZE; j < HALO_SIZE * 2; ++j) {
                ComputePoint(
                    workTempArrays[0], workTempArrays[1],
                    this->localDomainParamsGrid.data(),
                    this->localDomainMapGrid.data(),
                    i, j,
                    this->localGridRowSize,
                    floatConfig[AIR_FLOW],
                    floatConfig[COOLER_TEMP]
                );
            }

            for (size_t j = this->rightHeloPos - HALO_SIZE; j < this->rightHeloPos; ++j) {
                ComputePoint(
                    workTempArrays[0], workTempArrays[1],
                    this->localDomainParamsGrid.data(),
                    this->localDomainMapGrid.data(),
                    i, j,
                    this->localGridRowSize,
                    floatConfig[AIR_FLOW],
                    floatConfig[COOLER_TEMP]
                );
            }
        }

        //Start HALO XCHG
        reqCount = this->HaloXCHG(reqXCHG, workTempArrays[1]);

        UpdateTile(
            workTempArrays[0], workTempArrays[1],
            this->localDomainParamsGrid.data(),
            this->localDomainMapGrid.data(),
            HALO_SIZE * 2,
            HALO_SIZE * 2,
            this->localGridSizesWithHalo[X] - HALO_SIZE * 2,
            this->localGridSizesWithHalo[Y] - HALO_SIZE * 2,
            this->localGridRowSize,
            floatConfig[AIR_FLOW],
            floatConfig[COOLER_TEMP]
        );
        
        //Wait until XCHG is complete
        MPI_Waitall(reqCount, reqXCHG, MPI_STATUS_IGNORE);

        //save to file
        if (m_fileHandle != H5I_INVALID_HID && (iter % 10 == 0)) this->SaveToFile(workTempArrays[1], iter);

        //have I middle colum? compute temperature in mid col
        float globalSum = 0;
        if (this->middleCol != -1) {
            float localSum  = this->ComputeColSum(workTempArrays[1], this->middleCol);
            
            DEBUG_PRINT(MPI_ALL_RANKS, "My local temp sum is %f and global is %f\n", localSum, globalSum);
            MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, MPI_ROOT_RANK, this->MPIColComm);
            DEBUG_PRINT(MPI_COL_ROOT_RANK, "My global is %f from %d nodes\n", globalSum, this->localGridCounts[Y]);

            if (this->m_rank == MPI_COL_ROOT_RANK) {
                MPI_Send(&globalSum, 1, MPI_FLOAT, MPI_ROOT_RANK, NONE_TAG, this->MPIGridComm);
            }
        }

        std::swap(workTempArrays[0], workTempArrays[1]);

        if (this->m_rank == MPI_ROOT_RANK) {
            MPI_Recv(&globalSum, 1, MPI_FLOAT, this->middleColRootRank, NONE_TAG, this->MPIGridComm, MPI_STATUS_IGNORE);
            middleColAvgTemp = globalSum / float(m_materialProperties.GetEdgeSize());
            PrintProgressReport(iter, middleColAvgTemp);
        }

        // Wait until all sub arrays is done
        MPI_Barrier(this->MPIGridComm);
    }

    //print results
    if (this->m_rank == MPI_ROOT_RANK) {
        double elapsedTime = MPI_Wtime() - startTime;
        PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
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
