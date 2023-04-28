/**
 * @file    parallel_heat_solver.cpp
 * @author  xpleva07 <xpleva07@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-MM-DD
 */

#include "parallel_heat_solver.h"

using namespace std;

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

    DEBUG_PRINT(MPI_ROOT_RANK, "Init done\n");

}

ParallelHeatSolver::~ParallelHeatSolver()
{   
    MPI_Type_free(&this->MPILocalGridResized_T);
    MPI_Type_free(&this->MPILocalGridWithHalo_T);
    MPI_Type_free(&this->MPILocalINTGridWithHalo_T);
    MPI_Type_free(&this->MPILocalGridRow_T);
    MPI_Type_free(&this->MPILocalINTGridRow_T);
    MPI_Type_free(&this->MPILocalGridCol_T);
    MPI_Type_free(&this->MPILocalINTGridCol_T);

    MPI_Comm_free(&this->MPIColComm);
    MPI_Comm_free(&this->MPIGridComm);
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
    
    this->DiskWriteIntensity = this->m_simulationProperties.GetDiskWriteIntensity();

    MPI_Bcast(&this->DiskWriteIntensity, 1, MPI_SIZE_T, MPI_ROOT_RANK, MPI_COMM_WORLD);

    if (this->UseParallelIO) {
        DEBUG_PRINT(MPI_ALL_RANKS, "Setuping paraller IO to file name is %s write intensity is %d\n", this->FileName.c_str(), this->DiskWriteIntensity);
        
        // Create a property list to open the file using MPI-IO in the MPI_COMM_WORLD communicator.
        hid_t accesPList = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(accesPList, MPI_COMM_WORLD, MPI_INFO_NULL);

        // Create a file called (filename) with write permission. Use such a flag that overrides existing file.
        m_fileHandle.Set(H5Fcreate(this->FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, accesPList), H5Fclose);

        // Close file access list.
        H5Pclose(accesPList);
    } else if (this->m_rank == MPI_ROOT_RANK) {
        DEBUG_PRINT(MPI_ALL_RANKS, "Setuping seq IO to file %s write intensity is %d\n", this->FileName.c_str(), this->DiskWriteIntensity);
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
    this->downHeloPos       = TOTAL_SIZE_WITH_HALO(this->localGridSizes) - (this->localGridRowSize * (HALO_SIZE + (!this->neighbours[UP]  ? HALO_SIZE : 0))) - (this->neighbours[DOWN] ? 0 : HALO_SIZE);
    this->rightHeloPos      = haloStarts[X] + this->localGridSizes[X] - (this->neighbours[RIGHT] ? 0 : HALO_SIZE);

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

int ParallelHeatSolver::WindowHaloXCHG(MPI_Win &win, float* array) {
    if (this->neighbours[UP])
        MPI_Put(FIRST_LINE(array),   2, this->MPILocalGridRow_T, this->neighboursRanks[UP],    NEIGHBOUR_DOWN_HALO,   2, this->MPILocalGridRow_T, win);
    if (this->neighbours[DOWN])
        MPI_Put(LAST_2_LINES(array), 2, this->MPILocalGridRow_T, this->neighboursRanks[DOWN],  NEIGHBOUR_UP_HALO,     2, this->MPILocalGridRow_T, win);
    if (this->neighbours[LEFT])
        MPI_Put(FIRST_COL(array),    2, this->MPILocalGridCol_T, this->neighboursRanks[LEFT],  NEIGHBOUR_RIGHT_HALO,  2, this->MPILocalGridCol_T, win);
    if (this->neighbours[RIGHT])
        MPI_Put(LAST_2_COLS(array),  2, this->MPILocalGridCol_T, this->neighboursRanks[RIGHT], NEIGHBOUR_LEFT_HALO,   2, this->MPILocalGridCol_T, win);

    return 0;
}

float ParallelHeatSolver::ComputeColSum(const float *data, int index) {

    float tempSum = 0.0f;

    if (index < 0) return tempSum;

    for(size_t i = (this->neighbours[UP] ? HALO_SIZE : 0); i < this->localGridSizesWithHalo[Y] - (this->neighbours[DOWN] ? HALO_SIZE : 0); ++i) {
        tempSum += GET(index, i, data, this->localGridRowSize);
    }

    return tempSum;
}

void ParallelHeatSolver::removeHalos(const float *data, std::vector<float, AlignedAllocator<float>> &outData) {
    outData.reserve(TOTAL_SIZE(this->localGridSizes));

    size_t gridXStart = (this->neighbours[LEFT] ? HALO_SIZE : 0);
    size_t gridYStart = (this->neighbours[UP]   ? HALO_SIZE : 0);
    size_t gridXStop  = this->localGridSizesWithHalo[X] - (this->neighbours[RIGHT] ? HALO_SIZE : 0);
    size_t gridYStop  = this->localGridSizesWithHalo[Y] - (this->neighbours[DOWN]  ? HALO_SIZE : 0);

    size_t k = 0;
    for (size_t i = gridYStart;  i < gridYStop; ++i)  {
        for (size_t j = gridXStart; j < gridXStop; ++j) {
            outData[k++] = GET(j, i, data, this->localGridRowSize);
        }
    }
}

void ParallelHeatSolver::SaveToFile(const float *data, size_t iter) {
    if (this->UseParallelIO) {

        DEBUG_PRINT(MPI_ROOT_RANK, "Writing to file using paraller IO\n");

        // Create new HDF5 file group named as "Timestep_N", where "N" is number
        // of current snapshot. The group is placed into root of the file "/Timestep_N".
        std::string groupName = "Timestep_" + std::to_string(static_cast<unsigned long long>(iter / this->DiskWriteIntensity));
        AutoHandle<hid_t> groupHandle(H5Gcreate(this->m_fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);

        {
            hsize_t hGlobalGridSizes[] = {hsize_t(this->globalGridSizes[Y]), hsize_t(this->globalGridSizes[X])};
            hsize_t hLocalGridSizes[]  = {hsize_t(this->localGridSizes[Y]),  hsize_t(this->localGridSizes[X])};

            //rank is 2 we have 2D dimensional data
            hid_t fileSpace = H5Screate_simple(2, hGlobalGridSizes, nullptr);
            hid_t memSpace  = H5Screate_simple(2, hLocalGridSizes,  nullptr);

            std::string dataSetName("Temperature");

            // Create a dataset
            hid_t dataset = H5Dcreate(groupHandle, dataSetName.c_str(), H5T_NATIVE_FLOAT, fileSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            // Compute localgrid offset in 2D dataset
            hsize_t offset[] = {hsize_t(this->m_coords[Y] * this->localGridSizes[Y]), hsize_t(this->m_coords[X] * this->localGridSizes[X])};
            
            H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, offset, nullptr, hLocalGridSizes, nullptr);
            
            // Create XFER property list and set Collective IO.
            hid_t xferPList = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(xferPList, H5FD_MPIO_COLLECTIVE);

            // get local grid without HALOS
            std::vector<float, AlignedAllocator<float>> noHalosLocalGrid;
            this->removeHalos(data, noHalosLocalGrid);

            // Write data into the dataset.
            H5Dwrite(dataset, H5T_NATIVE_FLOAT, memSpace, fileSpace, xferPList, noHalosLocalGrid.data());
            
            // Close XREF property list.
            H5Pclose(xferPList);

            // Close spaces
            H5Sclose(memSpace);
            H5Sclose(fileSpace);
            
            // Close dataset.
            H5Dclose(dataset);
        }
        
        {
            // Create Integer attribute in the same group "/Timestep_N/Time"
            // in which we store number of current simulation iteration.
            std::string attributeName("Time");
            
            // Dataspace is single value/scalar.
            AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);
            
            // Create the attribute in the group as double.
            AutoHandle<hid_t> attributeHandle(H5Acreate2(groupHandle, attributeName.c_str(), H5T_IEEE_F64LE, dataSpaceHandle, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
                                                     
            // Write value into the attribute.
            double snapshotTime = double(iter);
            H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
        }

    } else {
        DEBUG_PRINT(MPI_ROOT_RANK, "Writing to file using serial IO\n");

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

        //save the data on root
        if (this->m_rank == MPI_ROOT_RANK) {
            StoreDataIntoFile(this->m_fileHandle, iter, outResult.data());
        }
    }
}

void ParallelHeatSolver::ComputeHalo(float **workTempArrays) {
    //first 2 lines
    for (size_t i = HALO_SIZE;  i < HALO_SIZE * 2; ++i)  {
        for (size_t j = HALO_SIZE; j < this->localGridSizesWithHalo[X] - HALO_SIZE; ++j) {
            ComputePoint(
                workTempArrays[0], workTempArrays[1],
                this->localDomainParamsGrid.data(),
                this->localDomainMapGrid.data(),
                i, j,
                this->localGridRowSize,
                this->floatConfig[AIR_FLOW],
                this->floatConfig[COOLER_TEMP]
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
                this->floatConfig[AIR_FLOW],
                this->floatConfig[COOLER_TEMP]
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
                this->floatConfig[AIR_FLOW],
                this->floatConfig[COOLER_TEMP]
            );
        }

        for (size_t j = this->rightHeloPos - HALO_SIZE; j < this->rightHeloPos; ++j) {
            ComputePoint(
                workTempArrays[0], workTempArrays[1],
                this->localDomainParamsGrid.data(),
                this->localDomainMapGrid.data(),
                i, j,
                this->localGridRowSize,
                this->floatConfig[AIR_FLOW],
                this->floatConfig[COOLER_TEMP]
            );
        }
    }
} 

void ParallelHeatSolver::SetupSolver() {
    //reserve space
    this->localTempGrid        .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localTempGrid2       .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localDomainParamsGrid.reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));
    this->localDomainMapGrid   .reserve(TOTAL_SIZE_WITH_HALO(this->localGridSizes));

    // set simulation configs
    floatConfig[AIR_FLOW]    = m_simulationProperties.GetAirFlowRate();
    floatConfig[COOLER_TEMP] = m_materialProperties.GetCoolerTemp();

    intConfig[ITER_NUM]      = m_simulationProperties.GetNumIterations();
    intConfig[RMA_MODE]      = m_simulationProperties.IsRunParallelRMA();

    // distribute simulation config between nodes
    MPI_Bcast(this->floatConfig, 2, MPI_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(this->intConfig,   2, MPI_INT,   MPI_ROOT_RANK, MPI_COMM_WORLD);

    DEBUG_PRINT(MPI_ALL_RANKS, "Is running in RMA mode: %d\n", intConfig[RMA_MODE]);

    //distribute inital data of material
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

    //distribute inital data of domain map
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

    //distribute inital data of init temp
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

    //exchage halozones material info between neighbours nodes 
    this->HaloMaterialXCHG();

    MPI_Request reqXCHG[HALO_REQ_COUNT];    
    int reqCount = this->HaloXCHG(reqXCHG, this->localTempGrid.data());

    MPI_Waitall(reqCount, reqXCHG, MPI_STATUS_IGNORE);

    //copy correctly to second array
    for (unsigned i = 0; i < TOTAL_SIZE_WITH_HALO(this->localGridSizes); i++) {
        this->localTempGrid2[i] = this->localTempGrid[i];
    }    

}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
    MPI_Win winODD;
    MPI_Win winEVEN;

    // ready simulator to start
    this->SetupSolver();

    float *workTempArrays[] = { this->localTempGrid.data(), this->localTempGrid2.data()};

    // MPI Window create
    if (intConfig[RMA_MODE]) {
        MPI_Win_create(workTempArrays[1], TOTAL_SIZE_WITH_HALO(this->localGridSizes) * sizeof(float), sizeof(float), MPI_INFO_NULL, this->MPIGridComm, &winEVEN);
        MPI_Win_create(workTempArrays[0], TOTAL_SIZE_WITH_HALO(this->localGridSizes) * sizeof(float), sizeof(float), MPI_INFO_NULL, this->MPIGridComm, &winODD);
        MPI_Win_fence(0, winODD);
        MPI_Win_fence(0, winEVEN);
    }

    // Note the start time
    double startTime = MPI_Wtime();

    float       middleColAvgTemp = 0;
    int         reqCount         = 0;
    MPI_Request reqXCHG[HALO_REQ_COUNT]; 

    // Begin iterative simulation main loop
    for(size_t iter = 0; iter < this->intConfig[ITER_NUM]; ++iter) {

        // first compute halo
        this->ComputeHalo(workTempArrays);

        // Start HALO XCHG
        if (intConfig[RMA_MODE])  reqCount = this->WindowHaloXCHG((iter % 2 == 0 ? winEVEN : winODD), workTempArrays[1]); //use RMA
        else                      reqCount = this->HaloXCHG(reqXCHG, workTempArrays[1]);   //use P2P

        // compute rest of array
        UpdateTile(
            workTempArrays[0], workTempArrays[1],
            this->localDomainParamsGrid.data(),
            this->localDomainMapGrid.data(),
            HALO_SIZE,
            HALO_SIZE,
            this->localGridSizesWithHalo[X] - HALO_SIZE * 2,
            this->localGridSizesWithHalo[Y] - HALO_SIZE * 2,
            this->localGridRowSize,
            floatConfig[AIR_FLOW],
            floatConfig[COOLER_TEMP]
        );
        
        // Wait until XCHG is complete
        if (intConfig[RMA_MODE])  MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, (iter % 2 == 0 ? winEVEN : winODD)); // no local store only PUT from others
        else                      MPI_Waitall(reqCount, reqXCHG, MPI_STATUS_IGNORE);

        // save to file
        if (this->FileNameLen && (iter % this->DiskWriteIntensity == 0)) this->SaveToFile(workTempArrays[1], iter);

        // have I middle colum? compute temperature in mid col
        float globalSum = 0;
        if (this->middleCol != -1) {
            
            float localSum  = this->ComputeColSum(workTempArrays[1], this->middleCol);
            MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, MPI_ROOT_RANK, this->MPIColComm);

            if (this->m_rank == MPI_COL_ROOT_RANK) {
                MPI_Send(&globalSum, 1, MPI_FLOAT, MPI_ROOT_RANK, NONE_TAG, this->MPIGridComm);
            }
        }

        // swap work array (old and new)
        std::swap(workTempArrays[0], workTempArrays[1]);

        // print progress on root
        if (this->m_rank == MPI_ROOT_RANK) {
            MPI_Recv(&globalSum, 1, MPI_FLOAT, this->middleColRootRank, NONE_TAG, this->MPIGridComm, MPI_STATUS_IGNORE);
            middleColAvgTemp = globalSum / float(m_materialProperties.GetEdgeSize());
            
            PrintProgressReport(iter, middleColAvgTemp);
        }

        // Wait until all sub arrays is done (Not important for beather progress)
        MPI_Barrier(this->MPIGridComm);
    }

    if (intConfig[RMA_MODE]) {
        MPI_Win_free(&winODD);
        MPI_Win_free(&winEVEN);
    }

    // print results
    if (this->m_rank == MPI_ROOT_RANK) {
        double elapsedTime = MPI_Wtime() - startTime;
        PrintFinalReport(elapsedTime, middleColAvgTemp, "par");
    }

    // copy final result to output
    MPI_Gatherv(
        workTempArrays[0],
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
