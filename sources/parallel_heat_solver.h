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

#define DEBUG
#if defined(DEBUG)
 #define DEBUG_PRINT(rank, fmt, args...) this->mpiPrintf(rank, "[DEBUG] [MPI_Rank %03d] %s:%d:%s(): " fmt, this->m_rank, __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(rank, fmt, args...) /* Don't do anything in release builds */
#endif

#define LOCAL_GRID_ON(X, Y) this->localGrid[Y * this->localGridXCount + X]
#define MPI_ROOT_RANK 0
#define MPI_ALL_RANKS -1

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{
    //============================================================================//
    //                            *** BEGIN: NOTE ***
    //
    // Modify this class declaration as needed.
    // This class needs to provide at least:
    // - Constructor which passes SimulationProperties and MaterialProperties
    //   to the base class. (see below)
    // - Implementation of RunSolver method. (see below)
    // 
    // It is strongly encouraged to define methods and member variables to improve 
    // readability of your code!
    //
    //                             *** END: NOTE ***
    //============================================================================//
    
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

    void Decompose();
    void mpiPrintf(int who, const char* __restrict__ format, ...);
    void CreateTypes();

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
    int m_coords[2];

    int localGridSizes[2];
    int localGridCounts[2];

    int globalGridSizes[2];

    std::vector<float, AlignedAllocator<float>> localGrid; 
    std::vector<int, AlignedAllocator<int>>     localGridDisplacement;
    std::vector<int, AlignedAllocator<int>>     localGridScatterCounts;

    MPI_Comm     MPIGridComm;
    MPI_Datatype MPILocalGrid_T;
    MPI_Datatype MPILocalGridResized_T;
};

#endif // PARALLEL_HEAT_SOLVER_H
