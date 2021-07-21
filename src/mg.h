// ************************************************************************
//
//          miniGhost: stencil computations with boundary exchange.
//                 Copyright (2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Richard F. Barrett (rfbarre@sandia.gov) or
//                    Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************

#ifndef _mg_h_
#define _mg_h_

#define MG_PARALLEL_TYPE_DEFAULT 0x0
#define MG_PARALLEL_TYPE_PTHREADS 0x1
#define MG_PARALLEL_TYPE_ARGOBOTS 0x2
#define MG_PARALLEL_TYPE_OPENMP_FOR 0x4

#define MG_ALIGNMENT 64

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "../config.h"

#include <mpi.h>

#ifdef _MG_ARGOBOTS
#define MG_PARALLEL_TYPE MG_PARALLEL_TYPE_ARGOBOTS
#elif defined(_MG_OPENMP)
#define MG_PARALLEL_TYPE MG_PARALLEL_TYPE_OPENMP_FOR
#elif defined(_MG_SERIAL)
#define MG_PARALLEL_TYPE MG_PARALLEL_TYPE_DEFAULT
#else
#define MG_PARALLEL_TYPE MG_PARALLEL_TYPE_PTHREADS
#endif

#if MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_ARGOBOTS
#include <abt.h>
typedef struct MG_thread_t {
    ABT_thread thread, waiter;
    void (*func)(void *);
    void *arg;
    int terminated;
} MG_thread_t;
#else
#include <pthread.h>
typedef struct MG_thread_t {
    pthread_t thread;
    void (*func)(void *);
    void *arg;
    int terminated, freed;
} MG_thread_t;
#endif

static inline int MG_Max(int a, int b)
{
    return a > b ? a : b;
}
static inline int MG_Min(int a, int b)
{
    return a > b ? b : a;
}
static inline int MG_Max3i(int a, int b, int c)
{
    return MG_Max(MG_Max(a, b), c);
}
static inline int MG_Min3i(int a, int b, int c)
{
    return MG_Min(MG_Min(a, b), c);
}
static inline int MG_Pow(int a, int b)
{
    int num = 1;
    for (int i = 0; i < b; i++)
        num *= a;
    return num;
}
static inline void MG_world_rank_to_pxpypz(int rank, int npx, int npy, int npz,
                                           int *px, int *py, int *pz)
{
    *px = rank % npx;
    *py = (rank / npx) % npy;
    *pz = (rank / npx) / npy;
}

static inline int MG_pxpypz_to_world_rank(int npx, int npy, int npz, int px,
                                          int py, int pz)
{
    return px + npx * (py + npy * pz);
}

static inline size_t MG_get_aligned(size_t val)
{
    return (val + MG_ALIGNMENT - 1) & (~(((size_t)MG_ALIGNMENT) - 1));
}

static inline uint32_t MG_fast_rand32(uint32_t *p_seed)
{
    // George Marsaglia, "Xorshift RNGs", Journal of Statistical Software,
    // Articles, 2003
    uint32_t seed = *p_seed;
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    *p_seed = seed;
    return seed;
}

void MG_Assert_impl(const char *cond, const char *file, int line);
void MG_Error_impl(const char *error_msg, const char *file, int line);
#define MG_Error(_error_msg)                                                   \
    do {                                                                       \
        MG_Error_impl(_error_msg, __FILE__, __LINE__);                         \
    } while (0)
#define MG_Assert(_cond)                                                       \
    do {                                                                       \
        if (!(_cond)) {                                                        \
            MG_Assert_impl(#_cond, __FILE__, __LINE__);                        \
        }                                                                      \
    } while (0)

#define GRID_INIT_VAL 0.0

#if defined _MG_DOUBLE
typedef double MG_REAL;
#define MG_COMM_REAL MPI_DOUBLE
#else
typedef float MG_REAL;
#define MG_COMM_REAL MPI_FLOAT
#endif

#define MG_ALIGNED __attribute__((aligned(MG_ALIGNMENT)))

#define FIVE 5.0          // For stencil computations.
#define SEVEN 7.0         // For stencil computations.
#define NINE 9.0          // For stencil computations.
#define TWENTY_SEVEN 27.0 // For stencil computations.

#define GIGA 1.0e9

#define MG_STENCIL_2D5PT 0
#define MG_STENCIL_2D9PT 1
#define MG_STENCIL_3D7PT 2
#define MG_STENCIL_3D27PT 3

#define BOUNDARY_X0 0x01
#define BOUNDARY_XN 0x02
#define BOUNDARY_Y0 0x04
#define BOUNDARY_YN 0x08
#define BOUNDARY_Z0 0x10
#define BOUNDARY_ZN 0x20
#define BOUNDARY_ALL                                                           \
    (BOUNDARY_X0 | BOUNDARY_XN | BOUNDARY_Y0 | BOUNDARY_YN | BOUNDARY_Z0 |     \
     BOUNDARY_ZN)

// Faces
#define NEIGHBOR_X0 (1 << 0)
#define NEIGHBOR_XN (1 << 1)
#define NEIGHBOR_Y0 (1 << 2)
#define NEIGHBOR_YN (1 << 3)
#define NEIGHBOR_Z0 (1 << 4)
#define NEIGHBOR_ZN (1 << 5)
// Lines
#define NEIGHBOR_X0Y0 (1 << 6)
#define NEIGHBOR_X0YN (1 << 7)
#define NEIGHBOR_XNY0 (1 << 8)
#define NEIGHBOR_XNYN (1 << 9)
#define NEIGHBOR_Y0Z0 (1 << 10)
#define NEIGHBOR_Y0ZN (1 << 11)
#define NEIGHBOR_YNZ0 (1 << 12)
#define NEIGHBOR_YNZN (1 << 13)
#define NEIGHBOR_Z0X0 (1 << 14)
#define NEIGHBOR_Z0XN (1 << 15)
#define NEIGHBOR_ZNX0 (1 << 16)
#define NEIGHBOR_ZNXN (1 << 17)
// Points
#define NEIGHBOR_X0Y0Z0 (1 << 18)
#define NEIGHBOR_X0Y0ZN (1 << 19)
#define NEIGHBOR_X0YNZ0 (1 << 20)
#define NEIGHBOR_X0YNZN (1 << 21)
#define NEIGHBOR_XNY0Z0 (1 << 22)
#define NEIGHBOR_XNY0ZN (1 << 23)
#define NEIGHBOR_XNYNZ0 (1 << 24)
#define NEIGHBOR_XNYNZN (1 << 25)
#define NEIGHBOR_ALL ((1 << 26) - 1)

#define NEIGHBOR_ANY_X0                                                        \
    (NEIGHBOR_X0 | NEIGHBOR_X0Y0 | NEIGHBOR_X0YN | NEIGHBOR_Z0X0 |             \
     NEIGHBOR_ZNX0 | NEIGHBOR_X0Y0Z0 | NEIGHBOR_X0Y0ZN | NEIGHBOR_X0YNZ0 |     \
     NEIGHBOR_X0YNZN)
#define NEIGHBOR_ANY_XN                                                        \
    (NEIGHBOR_XN | NEIGHBOR_XNY0 | NEIGHBOR_XNYN | NEIGHBOR_Z0XN |             \
     NEIGHBOR_ZNXN | NEIGHBOR_XNY0Z0 | NEIGHBOR_XNY0ZN | NEIGHBOR_XNYNZ0 |     \
     NEIGHBOR_XNYNZN)
#define NEIGHBOR_ANY_Y0                                                        \
    (NEIGHBOR_Y0 | NEIGHBOR_X0Y0 | NEIGHBOR_XNY0 | NEIGHBOR_Y0Z0 |             \
     NEIGHBOR_Y0ZN | NEIGHBOR_X0Y0Z0 | NEIGHBOR_X0Y0ZN | NEIGHBOR_XNY0Z0 |     \
     NEIGHBOR_XNY0ZN)
#define NEIGHBOR_ANY_YN                                                        \
    (NEIGHBOR_YN | NEIGHBOR_X0YN | NEIGHBOR_XNYN | NEIGHBOR_YNZ0 |             \
     NEIGHBOR_YNZN | NEIGHBOR_X0YNZ0 | NEIGHBOR_X0YNZN | NEIGHBOR_XNYNZ0 |     \
     NEIGHBOR_XNYNZN)
#define NEIGHBOR_ANY_Z0                                                        \
    (NEIGHBOR_Z0 | NEIGHBOR_Y0Z0 | NEIGHBOR_YNZ0 | NEIGHBOR_Z0X0 |             \
     NEIGHBOR_Z0XN | NEIGHBOR_X0Y0Z0 | NEIGHBOR_X0YNZ0 | NEIGHBOR_XNY0Z0 |     \
     NEIGHBOR_XNYNZ0)
#define NEIGHBOR_ANY_ZN                                                        \
    (NEIGHBOR_ZN | NEIGHBOR_Y0ZN | NEIGHBOR_YNZN | NEIGHBOR_ZNX0 |             \
     NEIGHBOR_ZNXN | NEIGHBOR_X0Y0ZN | NEIGHBOR_X0YNZN | NEIGHBOR_XNY0ZN |     \
     NEIGHBOR_XNYNZN)

static inline int MG_boundary_to_neighbors(int boundary_flag,
                                           int check_diagonal)
{
    int neighbor_flag = 0;
    if (boundary_flag & BOUNDARY_X0)
        neighbor_flag |= NEIGHBOR_X0;
    if (boundary_flag & BOUNDARY_XN)
        neighbor_flag |= NEIGHBOR_XN;
    if (boundary_flag & BOUNDARY_Y0)
        neighbor_flag |= NEIGHBOR_Y0;
    if (boundary_flag & BOUNDARY_YN)
        neighbor_flag |= NEIGHBOR_YN;
    if (boundary_flag & BOUNDARY_Z0)
        neighbor_flag |= NEIGHBOR_Z0;
    if (boundary_flag & BOUNDARY_ZN)
        neighbor_flag |= NEIGHBOR_ZN;
    if (check_diagonal) {
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_Y0))
            neighbor_flag |= NEIGHBOR_X0Y0;
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_YN))
            neighbor_flag |= NEIGHBOR_X0YN;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_Y0))
            neighbor_flag |= NEIGHBOR_XNY0;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_YN))
            neighbor_flag |= NEIGHBOR_XNYN;
        if ((boundary_flag & BOUNDARY_Y0) && (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_Y0Z0;
        if ((boundary_flag & BOUNDARY_Y0) && (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_Y0ZN;
        if ((boundary_flag & BOUNDARY_YN) && (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_YNZ0;
        if ((boundary_flag & BOUNDARY_YN) && (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_YNZN;
        if ((boundary_flag & BOUNDARY_Z0) && (boundary_flag & BOUNDARY_X0))
            neighbor_flag |= NEIGHBOR_Z0X0;
        if ((boundary_flag & BOUNDARY_Z0) && (boundary_flag & BOUNDARY_XN))
            neighbor_flag |= NEIGHBOR_Z0XN;
        if ((boundary_flag & BOUNDARY_ZN) && (boundary_flag & BOUNDARY_X0))
            neighbor_flag |= NEIGHBOR_ZNX0;
        if ((boundary_flag & BOUNDARY_ZN) && (boundary_flag & BOUNDARY_XN))
            neighbor_flag |= NEIGHBOR_ZNXN;
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_Y0) &&
            (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_X0Y0Z0;
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_Y0) &&
            (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_X0Y0ZN;
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_YN) &&
            (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_X0YNZ0;
        if ((boundary_flag & BOUNDARY_X0) && (boundary_flag & BOUNDARY_YN) &&
            (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_X0YNZN;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_Y0) &&
            (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_XNY0Z0;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_Y0) &&
            (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_XNY0ZN;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_YN) &&
            (boundary_flag & BOUNDARY_Z0))
            neighbor_flag |= NEIGHBOR_XNYNZ0;
        if ((boundary_flag & BOUNDARY_XN) && (boundary_flag & BOUNDARY_YN) &&
            (boundary_flag & BOUNDARY_ZN))
            neighbor_flag |= NEIGHBOR_XNYNZN;
    }
    return neighbor_flag;
}

static inline int MG_neighbor_to_boundary(int neighbor_flag)
{
    switch (neighbor_flag) {
        case NEIGHBOR_X0:
            return (BOUNDARY_X0);
        case NEIGHBOR_XN:
            return (BOUNDARY_XN);
        case NEIGHBOR_Y0:
            return (BOUNDARY_Y0);
        case NEIGHBOR_YN:
            return (BOUNDARY_YN);
        case NEIGHBOR_Z0:
            return (BOUNDARY_Z0);
        case NEIGHBOR_ZN:
            return (BOUNDARY_ZN);
        case NEIGHBOR_X0Y0:
            return (BOUNDARY_X0 | BOUNDARY_Y0);
        case NEIGHBOR_X0YN:
            return (BOUNDARY_X0 | BOUNDARY_YN);
        case NEIGHBOR_XNY0:
            return (BOUNDARY_XN | BOUNDARY_Y0);
        case NEIGHBOR_XNYN:
            return (BOUNDARY_XN | BOUNDARY_YN);
        case NEIGHBOR_Y0Z0:
            return (BOUNDARY_Y0 | BOUNDARY_Z0);
        case NEIGHBOR_Y0ZN:
            return (BOUNDARY_Y0 | BOUNDARY_ZN);
        case NEIGHBOR_YNZ0:
            return (BOUNDARY_YN | BOUNDARY_Z0);
        case NEIGHBOR_YNZN:
            return (BOUNDARY_YN | BOUNDARY_ZN);
        case NEIGHBOR_Z0X0:
            return (BOUNDARY_Z0 | BOUNDARY_X0);
        case NEIGHBOR_Z0XN:
            return (BOUNDARY_Z0 | BOUNDARY_XN);
        case NEIGHBOR_ZNX0:
            return (BOUNDARY_ZN | BOUNDARY_X0);
        case NEIGHBOR_ZNXN:
            return (BOUNDARY_ZN | BOUNDARY_XN);
        case NEIGHBOR_X0Y0Z0:
            return (BOUNDARY_X0 | BOUNDARY_Y0 | BOUNDARY_Z0);
        case NEIGHBOR_X0Y0ZN:
            return (BOUNDARY_X0 | BOUNDARY_Y0 | BOUNDARY_ZN);
        case NEIGHBOR_X0YNZ0:
            return (BOUNDARY_X0 | BOUNDARY_YN | BOUNDARY_Z0);
        case NEIGHBOR_X0YNZN:
            return (BOUNDARY_X0 | BOUNDARY_YN | BOUNDARY_ZN);
        case NEIGHBOR_XNY0Z0:
            return (BOUNDARY_XN | BOUNDARY_Y0 | BOUNDARY_Z0);
        case NEIGHBOR_XNY0ZN:
            return (BOUNDARY_XN | BOUNDARY_Y0 | BOUNDARY_ZN);
        case NEIGHBOR_XNYNZ0:
            return (BOUNDARY_XN | BOUNDARY_YN | BOUNDARY_Z0);
        case NEIGHBOR_XNYNZN:
            return (BOUNDARY_XN | BOUNDARY_YN | BOUNDARY_ZN);
        default:
            MG_Assert(0);
    }
    return -1;
}

static inline int MG_neighbor_to_boundaries(int neighbor_flag)
{
    int boundary_flag = 0;
    if (neighbor_flag & NEIGHBOR_ANY_X0)
        boundary_flag |= BOUNDARY_X0;
    if (neighbor_flag & NEIGHBOR_ANY_XN)
        boundary_flag |= BOUNDARY_XN;
    if (neighbor_flag & NEIGHBOR_ANY_Y0)
        boundary_flag |= BOUNDARY_Y0;
    if (neighbor_flag & NEIGHBOR_ANY_YN)
        boundary_flag |= BOUNDARY_YN;
    if (neighbor_flag & NEIGHBOR_ANY_Z0)
        boundary_flag |= BOUNDARY_Z0;
    if (neighbor_flag & NEIGHBOR_ANY_ZN)
        boundary_flag |= BOUNDARY_ZN;
    return boundary_flag;
}

#define ADJACENTS_2D5PT                                                        \
    {                                                                          \
        NEIGHBOR_X0, NEIGHBOR_XN, NEIGHBOR_Y0, NEIGHBOR_YN                     \
    }
#define ADJACENTS_2D9PT                                                        \
    {                                                                          \
        NEIGHBOR_X0, NEIGHBOR_XN, NEIGHBOR_Y0, NEIGHBOR_YN, NEIGHBOR_X0Y0,     \
            NEIGHBOR_X0YN, NEIGHBOR_XNY0, NEIGHBOR_XNYN                        \
    }
#define ADJACENTS_3D7PT                                                        \
    {                                                                          \
        NEIGHBOR_X0, NEIGHBOR_XN, NEIGHBOR_Y0, NEIGHBOR_YN, NEIGHBOR_Z0,       \
            NEIGHBOR_ZN                                                        \
    }
#define ADJACENTS_3D27PT                                                       \
    {                                                                          \
        NEIGHBOR_X0, NEIGHBOR_XN, NEIGHBOR_Y0, NEIGHBOR_YN, NEIGHBOR_Z0,       \
            NEIGHBOR_ZN, NEIGHBOR_X0Y0, NEIGHBOR_X0YN, NEIGHBOR_XNY0,          \
            NEIGHBOR_XNYN, NEIGHBOR_Y0Z0, NEIGHBOR_Y0ZN, NEIGHBOR_YNZ0,        \
            NEIGHBOR_YNZN, NEIGHBOR_Z0X0, NEIGHBOR_Z0XN, NEIGHBOR_ZNX0,        \
            NEIGHBOR_ZNXN, NEIGHBOR_X0Y0Z0, NEIGHBOR_X0Y0ZN, NEIGHBOR_X0YNZ0,  \
            NEIGHBOR_X0YNZN, NEIGHBOR_XNY0Z0, NEIGHBOR_XNY0ZN,                 \
            NEIGHBOR_XNYNZ0, NEIGHBOR_XNYNZN                                   \
    }

static inline void MG_neighbor_to_sxsysz(int neighbor, int *x, int *y, int *z)
{
    int boundary = MG_neighbor_to_boundary(neighbor);
    *x = (boundary & BOUNDARY_XN) ? 1 : ((boundary & BOUNDARY_X0) ? -1 : 0);
    *y = (boundary & BOUNDARY_YN) ? 1 : ((boundary & BOUNDARY_Y0) ? -1 : 0);
    *z = (boundary & BOUNDARY_ZN) ? 1 : ((boundary & BOUNDARY_Z0) ? -1 : 0);
}

#define VALIDATE_TYPE_NONE 0x0
#define VALIDATE_TYPE_FLUX 0x1
#define VALIDATE_TYPE_VALUE 0x2
#define VALIDATE_TYPE_PRINT 0x4

// For indexing convenience:
#define MG_ARRAY_INDEX(i, j, k, w, h)                                          \
    ((((w) + 2) * (((h) + 2) * (k) + (j))) + (i))

typedef struct BlockInfo BlockInfo;
typedef struct GridInfo GridInfo;
typedef struct Params Params;

typedef struct CommInfo {
    MG_REAL *buffer;   // send-recv buffer.
    int rank;          // MPI rank
    int tag;           // MPI Tag
    MPI_Comm comm;     // Communicator.
    int buffer_len;    // Buffer length
    int neighbor_flag; // NEIGHBOR_XXX
} CommInfo;

typedef struct SyncInfo {
    int index;        // This owner of SyncInfo must be p_blk->syncs[index]
    BlockInfo *p_blk; // Neighbor block.
#if MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_ARGOBOTS
    ABT_thread thread;
#elif MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_PTHREADS
    int resume_flag;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
#endif
} SyncInfo;

typedef struct StencilArg {
    const Params *p_params;
    BlockInfo *p_blk;
    int num_steps;
} StencilArg;

#define ITER_SYNC_ITER_INC ((uint64_t)1 << (uint64_t)32)
#define ITER_SYNC_GET_SYNC(var) ((var) & (uint64_t)(ITER_SYNC_ITER_INC - 1))
#define ITER_SYNC_GET_ITER(var) ((var) >> (uint64_t)32)
#define ITER_SYNC_GET_ITER_SYNC(iter, sync)                                    \
    ((((uint64_t)iter) << ((uint64_t)32)) | (sync))

typedef MG_ALIGNED struct BlockInfo {
    // Almost constant values.
    int xstart, xend, ystart, yend, zstart, zend;
    int neighbor_flag;   // Flag of neighbors regarding communication.
    int boundary_flag;   // Physical boundary condition flag.
    int wraparound_flag; // Local warparound flag (for periodic condition).
    int num_comms;       // First buffers: send / Later buffers: recv.
    CommInfo *comms;     // Communicators this block needs to use.
    int num_syncs;       // # of synchronizations
    SyncInfo *syncs;     // Intra-process synchronization.
    GridInfo *p_grid;    // Grid information to which this block belongs.
    MG_REAL flux;        // Flux out of this block
    int iter;            // Iteration.
                         // iter % 2 == 0: values1 -> values2
                         // iter % 2 == 1: values2 -> values1

    // Used for threading.
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
    StencilArg arg;
    MG_thread_t thread;
#endif

    // Dynamic values that are accessed by other threads.
    MG_ALIGNED
    // Iteration counter.  This is updated in a very complicated manner.
    // The first 32 bits (actually 26 bits at maximum) are used for
    // synchronization: threads that wait for this block will wait on this
    // variable.  The remaining 32 bits are used to manage a real iteration.
    //
    // If there's no dependency, iter_sync is incremented by 1 << 32 after every
    // iteration by the owner of this block.  The owner needs to wake up all
    // the corresponding threads if the first 32 bits of iter_sync is not 0.
    //
    // Waiter: If a thread wants to wait on this variable, the thread uses
    // compare-and-swap this variable to set a flag to a corresponding bit.
    //
    // This iteration value is different from iter because this iteration value
    // ensures that this block AND its halo obtained by the communication have
    // already been updated.  Currently, the following relationship holds.
    // iter <= ITER_SYNC_GET_ITER(iter_sync) <= iter + 1.
    //
    // iteration value: ITER_SYNC_GET_ITER(iter_sync)
    // sync bits      : ITER_SYNC_GET_SYNC(iter_sync)
    //
    uint64_t iter_sync;

    // Unused values in the performance-critical paths.
    int blkx;         // Block x-position in local grid.
    int blky;         // Block y-position in local grid.
    int blkz;         // Block z-position in local grid.
    int id;           // Block ID,  This is used for communicator assignment.
    void *mem_buffer; // Used for comms and comms' buffer.
} BlockInfo;

typedef struct DebugGridInfo {
    int gnx;            // # of global cells, x-dim (nx * npx)
    int gny;            // # of global cells, y-dim (ny * npy)
    int gnz;            // # of global cells, z-dim (nz * npz)
    MG_REAL **p_values; // values[i][MG_ARRAY_INDEX(...)]: an i-th grid answer.
                        // Size of values[i]: (gnx + 2) * (gny + 2) * (gnz + 2)
    void *buffer;
} DebugGridInfo;

typedef MG_ALIGNED struct GridInfo {
    //  Alternate values between these two, with other serving as workspace.
    MG_REAL *values1;
    MG_REAL *values2;

    // Parallelization
    BlockInfo *blks; // Blocks[p_params->nblks] to compute those grids.

    // Debug/verification variables and parameters.
    int check_answer;
    MG_REAL source_total; // Source inputs across all the processes.

    // Grid values for debugging.
    DebugGridInfo *p_debug;
} GridInfo;

#define MG_EXEC_TYPE_COMM 0x1
#define MG_EXEC_TYPE_COMP 0x2
#define MG_EXEC_TYPE_NORMAL 0x3
#define MG_EXEC_TYPE_COPYCOMP 0x4

typedef struct Params {
    // Block information
    int nblksx; // Total number of blocks in x direction.
    int nblksy; // Total number of blocks in y direction.
    int nblksz; // Total number of blocks in z direction.

    // Per-process cell information
    int nx; // # of local cells, x-dim
    int ny; // # of local cells, y-dim
    int nz; // # of local cells, z-dim

    // Process information
    int npx; // Size of logical processor grid, x-dir
    int npy; // Size of logical processor grid, y-dir
    int npz; // Size of logical processor grid, z-dir

    // Configurations: parallelization
    int num_comms;     // Number of communicators.
    int comm_strategy; // MPI send/recv strategy.

    // Configurations: stencil problem
    int stencil;     // Stencil to be applied
    int numvars;     // Number of variables to be operated on.
    int numtsteps;   // Number of time steps per heat spike.
    int bc_periodic; // Boundary condition (1: periodic, 0: Dirichlet)

    // Configuration: debugging and verification options
    MG_REAL error_tol;     // Error tolerance
    int check_answer_freq; // Check answer every check_answer_freq time steps.
    int validation_type;   // 0: no validation
                           // 1: flux check
                           // 2: flux and grid value check
                           // 3: flux and grid value check + print
    int print_header;      // Print a header

    // Execution-time parameters.
    int nprocs; // Number of size
    int rank;   // Global rank
    int px; // Process grid ID (0 <= px < npx).  See MG_world_rank_to_pxpypz()
    int py; // Process grid ID (0 <= py < npy)
    int pz; // Process grid ID (0 <= pz < npz)
    int nblks;       // nblksx * nblksy * nblksz
    int ncblkx;      // Number of cells in each block, x-dir  (= nx / nblksx)
    int ncblky;      // Number of cells in each block, y-dir  (= ny / nblksy)
    int ncblkz;      // Number of cells in each block, z-dir  (= nz / nblksz)
    MPI_Comm *comms; // Communicators for neighbor communication.
                     // Length: num_comms
    int exec_type;   // Excecution Type
} Params;

// Those values are for debugging purpose.  Use Params.rank and nprocs if
// possible since accessing the following can pollute the cache.
extern int g_debug_rank;
extern int g_debug_nprocs;

void MG_Init(int argc, char *argv[], Params *p_params);
void MG_Finalize(Params *p_params);
void MG_Block_init(Params *p_params, BlockInfo **p_blks);
void MG_Block_finalize(Params *p_params, BlockInfo **p_blks);
void MG_Grid_init(const Params *p_params, GridInfo *p_grid);
void MG_Grid_finalize(const Params *p_params, GridInfo *p_grid);

void MG_Stencil(const Params *p_params, BlockInfo *p_blk);
void MG_Stencil_kernel(const MG_REAL *restrict grid_in,
                       MG_REAL *restrict grid_out, int nx, int ny, int nz,
                       int xstart, int xend, int ystart, int yend, int zstart,
                       int zend, int stencil_type);
void MG_Stencil_single(const Params *p_params, BlockInfo *blks, int num_blks,
                       int num_steps);
void MG_Stencil_thread(void *arg);

MG_REAL MG_Process_boundary_conditions(const Params *p_params,
                                       BlockInfo *p_blk);
void MG_Boundary_exchange(const Params *p_params, BlockInfo *p_blk);
void MG_Wraparound(const Params *p_params, int wraparound_flag,
                   MG_REAL *restrict grid, int nx, int ny, int nz, int xstart,
                   int xend, int ystart, int yend, int zstart, int zend);
void MG_Wraparound_blk(const Params *p_params, const BlockInfo *p_blk,
                       MG_REAL *restrict grid);

void MG_Pack_buffer(const Params *p_params, MG_REAL *grid, BlockInfo *p_blk);
void MG_Unpack_buffer(const Params *p_params, MG_REAL *grid, BlockInfo *p_blk);
void MG_Validate_results(const Params *p_params, const GridInfo *p_grid,
                         int step);

#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
void MG_Thread_session(void (*func)(void *), void *arg);
void MG_Thread_init_sync(SyncInfo *p_sync);
void MG_Thread_finalize_sync(SyncInfo *p_sync);
void MG_Thread_self_suspend(SyncInfo *p_sync);
void MG_Thread_resume(SyncInfo *p_sync);
void MG_Thread_create(void (*func)(void *), void *arg, MG_thread_t *p_thread);
void MG_Thread_revive(MG_thread_t *p_thread);
void MG_Thread_join(MG_thread_t *p_thread);
void MG_Thread_free(MG_thread_t *p_thread);
#endif

void MG_Print_header(const Params *p_params);
// Allocated memory can be released by "free()"
void *MG_malloc(size_t size);
void *MG_calloc(size_t size1, size_t size2);
#endif /* _mg_h_ */
