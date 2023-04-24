/**
 * @file    parallel_heat_solver.h
 * @author  xlogin00 <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2021/2022 - Project 1
 *
 * @date    2022-MM-DD
 */

#ifndef PARALLEL_HEAT_SOLVER_H
#define PARALLEL_HEAT_SOLVER_H

#include "base_heat_solver.h"
#include <stdint.h>
#include <limits.h>

//#define DEBUG
#if defined(DEBUG)
 #define DEBUG_PRINT(rank, fmt, args...) this->mpiPrintf(rank, "[DEBUG] [MPI_Rank %03d] %s:%d:%s(): " fmt, this->m_rank, __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(rank, fmt, args...) /* Don't do anything in release builds */
#endif

#define LOCAL_GRID_ON(X, Y) this->localGrid[Y * this->localGridXCount + X]
#define MPI_ROOT_RANK 0
#define MPI_ALL_RANKS -1
#define MPI_COL_ROOT_RANK this->m_coords[X]

#define X 1
#define Y 0
#define HALO_SIZE 2

#define UP        0
#define DOWN      1
#define LEFT      2
#define RIGHT     3

#define COOLER_TEMP 1
#define AIR_FLOW    0

#define ITER_NUM    0

#define HALO_REQ_COUNT 8

#define NONE_TAG 0

#define UP_HALO(A)      &(A[0])
#define DOWN_HALO(A)    &(A[this->downHeloPos])
#define LEFT_HALO(A)    &(A[0])
#define RIGHT_HALO(A)   &(A[this->rightHeloPos])

#define FIRST_LINE(A)   &(A[this->dataStartPos])
#define LAST_2_LINES(A) &(A[this->downHeloPos - this->localGridRowSize * 2])
#define FIRST_COL(A)    &(A[this->dataStartColPos])
#define LAST_2_COLS(A)  &(A[this->rightHeloPos - 2])

#define NEIGHBOUR_UP_HALO      0
#define NEIGHBOUR_DOWN_HALO    (this->localGridRowSize * ((this->m_coords[Y] > 1 ? HALO_SIZE : 0) + this->localGridSizes[Y]))
#define NEIGHBOUR_LEFT_HALO    0
#define NEIGHBOUR_RIGHT_HALO   (this->localGridSizes[X] + (this->m_coords[X] > 1 ? HALO_SIZE : 0))

#define TOTAL_SIZE(A) (A[X] * A[Y])
#define TOTAL_SIZE_WITH_HALO(A) ((A[X] + 2 * HALO_SIZE) * (A[Y] + 2 * HALO_SIZE))
#define GET(X_POS, Y_POS, ARRAY, ROW_SIZE) ARRAY[(Y_POS * ROW_SIZE + X_POS)]
#define LOCAL_GET(X_POS, Y_POS, ARRAY) GET(X_POS, Y_POS, ARRAY, this->localGridSizes[X])
#define LOCAL_HALO_GET(X_POS, Y_POS, ARRAY) GET(X_POS, Y_POS, ARRAY, (this->localGridSizes[X] + 2 * HALO_SIZE))

#if SIZE_MAX == UCHAR_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "unable to determinate SIZE_T bit width"
#endif

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{    
public:
    /**
     * @brief Constructor - Initializes the solver. This should include things like:
     *        - Construct 1D or 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tile.
     *        - Initialize persistent communications?
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);
    virtual ~ParallelHeatSolver();

    /**
     * Do 1D and 2D decomposition on MPI rank create new CART topology comunicator
     * compute local, global array sizes and distribute this info between ranks
     * compute neighbours ranks and create Column comunicator
     */
    void Decompose();

    /**
     * C printf routine with selection which rank prints.
     * @param who    - which rank should print. If -1 then all prints.
     * @param format - format string.
     * @param ...    - other parameters.
     */
    void mpiPrintf(int who, const char* __restrict__ format, ...);

    /**
     * Create MPI DATA TYPES and cumpute useful array indexes
     * DATA_TYPES:
     *      MPI_Datatype MPILocalGridWithHalo_T;
     *      MPI_Datatype MPILocalGridResized_T;
     *      MPI_Datatype MPILocalINTGridWithHalo_T;
     *      MPI_Datatype MPILocalGridRow_T;
     *      MPI_Datatype MPILocalINTGridRow_T;
     *      MPI_Datatype MPILocalGridCol_T;
     *      MPI_Datatype MPILocalINTGridCol_T;
     * INDEXES:
     *      localGridDisplacement
     *      localGridScatterCounts
     *      localGridSizesWithHalo
     *      middleCol
     *      middleColRootRank
     *      localGridRowSize
     *      localGridColSize
     *      dataStartPos
     *      dataStartColPos
     *      downHeloPos
     *      rightHeloPos
     */
    void CreateTypes();

    /**
     * Exchage halo regeions of floats with neighbours
     * @param req   pointer to array to save MPI_Request
     * @param array array with halos to exchage
     * @post  need to wait for all req in req after in 
     *        halozones is actual numbers from neighbours
     *        and neighbours have actual numbers from array
     *        in theiers halo zones
     * @return number of req
     */
    int  HaloXCHG(MPI_Request* req, float* array);

    /**
     * Exchage halo regeions of ints with neighbours
     * @param req   pointer to array to save MPI_Request
     * @param array array with halos to exchage
     * @post  need to wait for all req in req after in 
     *        halozones is actual numbers from neighbours
     *        and neighbours have actual numbers from array
     *        in theiers halo zones
     * @return number of req
     */
    int  HaloINTXCHG(MPI_Request* req, int* array);

    /**
     * Exchage halo material info regeions with neighbours
     * Exchage this->localDomainParamsGrid and this->localDomainMapGrid
     */
    void HaloMaterialXCHG();

    /**
     * Compute sum of points in specific col
     * @param data pointer to array with col
     * @param index index of start of column
     * @return sum of column
     */
    float ComputeColSum(const float *data, int index);

    /**
     * Init HDF5 file handler and exchage all important information about file
     * Setuped vars:
     *  FileNameLen
     *  FileName
     *  UseParallelIO
     *  DiskWriteIntensity
     *  m_fileHandle
     */
    void ReserveFile();

    /**
     * Save state of current iteration to file
     * auto use seq or par IO mode by this->UseParallelIO
     * @param data current state to save
     * @param iter current iteraton number
     */
    void SaveToFile(const float *data, size_t iter);

    /**
     * Compute points whats is in neighbours halo zones
     * @param workTempArrays pointer to work array (array of two temperature arrays)
     * @post in workTempArrays[NEW] in points whats is in neighbours halo zones is computed new values
     */
    void ComputeHalo(float **workTempArrays);

    /**
     * Setup simulation for start
     * Distribute all needed data between all ranks using scatter and bcast
     * Distributed:
     *  intConfig              BCAST
     *  floatConfig            BCAST
     *  localTempGrid          SCATTER from GetInitTemp()
     *  localTempGrid2         SCATTER from GetInitTemp()
     *  localDomainParamsGrid  SCATTER from GetDomainParams()
     *  localDomainMapGrid     SCATTER form GetDomainMap()
     */
    void SetupSolver();

    /**
     * Create new grid with removed hallo from original
     * @param data pointer to grid with halo zones
     * @param outData target vector to save new grid without haloZones
     */
    void removeHalos(const float *data, std::vector<float, AlignedAllocator<float>> &outData);

    int WindowHaloXCHG(MPI_Win &win, float* array);

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

protected:
    int m_rank;     ///< Process rank in global (MPI_COMM_WORLD) communicator.
    int m_size;     ///< Total number of processes in MPI_COMM_WORLD.
    int m_coords[2]; ///< position in MPI_CART topologi

    int localGridSizes[2]; ///< X Y Size of local array
    int localGridCounts[2]; ///< X Y Count of ranks in MPI_CART
    int localGridSizesWithHalo[2]; ///< X Y size of data in local array counted with halo zones
    float floatConfig[2]; ///< simulation float prameters like air flow and cooler temperature
    int   intConfig  [2]; ///< simulation int parameter like number of interations

    bool neighbours[4] = {false, false, false, false}; ///< existence of LEFT RIGHT UP DOWN neighbours
    int  neighboursRanks[4]; ///< rank ids of LEFT RIGHT UP DOWN neighbours
    
    int localGridRowSize; ///< total size of row in local grid with not used points and halos
    int localGridColSize; ///< total size of col in local grid with not used points and halos
    int dataStartPos;     ///< index of start of data in grid (skip HALOS)
    int downHeloPos;      ///< index of start of down helo
    int rightHeloPos;     ///< index of start of right helo
    int dataStartColPos;  ///< index of start of data by column indexing (skip LEFT HALO)
    int middleCol;        ///< index of absolute middle column in local array if not present -1
    int middleColRootRank;///< index of root rank of middle column (TOP RANK of column)
    int FileNameLen;      ///< lenght of output file name 

    bool UseParallelIO = false; ///< use parallel IO
    size_t DiskWriteIntensity; ///< every n interation write to file

    std::string FileName; ///< name of output file

    int globalGridSizes[2]; ///< X Y sizes of global grid

    std::vector<float, AlignedAllocator<float>> localTempGrid; ///< local temp grid
    std::vector<float, AlignedAllocator<float>> localTempGrid2; ///< local temp grid 2 (for temp work - old and new)
    std::vector<int, AlignedAllocator<int>>     localGridDisplacement; ///< displcament of local grids in global grid
    std::vector<int, AlignedAllocator<int>>     localGridScatterCounts; ///< numbes of local grids onranks (all only 1) 

    std::vector<float, AlignedAllocator<float>> localDomainParamsGrid; ///< array with local Domain params
    std::vector<int, AlignedAllocator<int>>     localDomainMapGrid; ///< array with local domian map

    MPI_Comm     MPIGridComm; ///< CART grid comunicator
    MPI_Comm     MPIColComm; ///< comunicator for column
    MPI_Datatype MPILocalGridWithHalo_T; ///< Type of local grid with addition halo zone
    MPI_Datatype MPILocalGridResized_T; ///< Type of local grid in global grid
    MPI_Datatype MPILocalINTGridWithHalo_T; ///< Type of local grid with addition halo zone
    MPI_Datatype MPILocalGridRow_T; ///< Type for local grid row
    MPI_Datatype MPILocalINTGridRow_T;///< Type for local grid row
    MPI_Datatype MPILocalGridCol_T; ///< Type for local grid column
    MPI_Datatype MPILocalINTGridCol_T;///< Type for local grid column

    AutoHandle<hid_t> m_fileHandle;                             ///< Output HDF5 file handle.
};

#endif // PARALLEL_HEAT_SOLVER_H
