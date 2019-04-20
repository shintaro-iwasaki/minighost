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

#include "mg.h"

void MG_Stencil_single(const Params *p_params, BlockInfo *blks, int num_blks,
                       int num_steps)
{
    const int exec_type = p_params->exec_type;
    for (int step = 0; step < num_steps; step++) {
        if (exec_type & MG_EXEC_TYPE_COMM) {
            // Boundary exchange.  All exchanges must be done first.
            if (exec_type & MG_EXEC_TYPE_COMM) {
                for (int blkid = 0; blkid < num_blks; blkid++) {
                    BlockInfo *p_blk = &blks[blkid];
                    MG_Boundary_exchange(p_params, p_blk);
                }
            }
        }
        // Execute compute kernels.  No local synchronization is needed.
        if (exec_type & MG_EXEC_TYPE_COMP) {
            for (int blkid = 0; blkid < num_blks; blkid++) {
                BlockInfo *p_blk = &blks[blkid];
                // Calculate the boundary.
                p_blk->flux += MG_Process_boundary_conditions(p_params, p_blk);
                // Apply stencil.
                MG_Stencil(p_params, p_blk);
            }
        }
        // Update the iteration.
        for (int blkid = 0; blkid < num_blks; blkid++) {
            BlockInfo *p_blk = &blks[blkid];
            p_blk->iter++;
        }
    }
}

void MG_Stencil_thread(void *arg)
{
    StencilArg *p_arg = (StencilArg *)arg;
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
    const Params *p_params = p_arg->p_params;
    BlockInfo *p_blk = p_arg->p_blk;
    const int num_steps = p_arg->num_steps;
    const int exec_type = p_params->exec_type;
#if MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_PTHREADS
    // Set up affinity.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(p_blk->id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    for (int step = 0; step < num_steps; step++) {

        if (exec_type & MG_EXEC_TYPE_COMM) {
            // Exchange the boundary.  Now this block and its halo are updated.
            MG_Boundary_exchange(p_params, p_blk);
            // For a correct halo, it needs to take care of the following
            // dependencies.
            // 1. To perform the t computation, all the other blocks must have
            //    finished the t - 1 computation and obtained a halo of t - 1
            //    computation (if needed).
            // 2. While the t computation is being performed, all the other
            // blocks
            //    may not start the t + 1 computation (otherwise the
            //    double-buffered grid is overwritten).
            // The second condition is satisfied if the first condition is
            // satisfied.
            //
            // The first condition is checked in this function.  Completion
            // signals are issued right after the stencil kernel.
            //
            // Here the t -1 boundary exchange (as well as the previous t - 1
            // computation) has been done.  Once this iter_sync's first 32 bit
            // is updated, the old value could be overwritten by other
            // neighboring blocks.
            uint64_t my_iter_sync;
            {
                uint64_t old_iter_sync;
                while (1) {
                    // Need to know exactly which threads need to be resumed.
                    // Let's use an atomic operation.
                    old_iter_sync =
                        __atomic_load_n(&p_blk->iter_sync, __ATOMIC_ACQUIRE);
                    uint64_t expected_val =
                        ITER_SYNC_GET_ITER_SYNC(ITER_SYNC_GET_ITER(
                                                    old_iter_sync) +
                                                    1,
                                                0);
                    if (__atomic_compare_exchange_n(&p_blk->iter_sync,
                                                    &old_iter_sync,
                                                    expected_val, 1,
                                                    __ATOMIC_ACQ_REL,
                                                    __ATOMIC_ACQUIRE)) {
                        // Succeeded.
                        my_iter_sync = expected_val;
                        break;
                    }
                }
                // Need to wake up threads that are waiting
                for (int i = 0, ie = p_blk->num_syncs; i < ie; i++) {
                    if (old_iter_sync & ((uint64_t)1 << (uint64_t)i)) {
                        MG_Thread_resume(&p_blk->syncs[i]);
                    }
                }
            }
            for (int i = 0, ie = p_blk->num_syncs; i < ie; i++) {
                SyncInfo *p_sync = &p_blk->syncs[i];
                // Quick check path.
                uint64_t iter_sync = __atomic_load_n(&p_sync->p_blk->iter_sync,
                                                     __ATOMIC_ACQUIRE);
                while (ITER_SYNC_GET_ITER(iter_sync) <
                       ITER_SYNC_GET_ITER(my_iter_sync)) {
                    // Double check the value by using CAS.  Assuming that
                    // iter_sync is unchanged.
                    uint64_t expected_val =
                        iter_sync | ((uint64_t)1 << (uint64_t)p_sync->index);
                    if (__atomic_compare_exchange_n(&p_sync->p_blk->iter_sync,
                                                    &iter_sync, expected_val, 1,
                                                    __ATOMIC_ACQ_REL,
                                                    __ATOMIC_ACQUIRE)) {
                        // CAS succeeded.  It means that the t - 1 computation
                        // has not finished.  Let's sleep this thread.  Since
                        // the stencil kernel also uses an atomic operation,
                        // that thread is aware of this waiter.
                        MG_Thread_self_suspend(
                            &p_sync->p_blk->syncs[p_sync->index]);
                        // Argobots does not need it, but let's consider a
                        // spurious wakeup too.  Enter the loop again.
                    }
                    // CAS failed.  It seems that the value is updated by
                    // someone else. Let's check the latest value.
                    iter_sync = __atomic_load_n(&p_sync->p_blk->iter_sync,
                                                __ATOMIC_ACQUIRE);
                }
                // t - 1 computation has finished.
            }
        }
        if (exec_type & MG_EXEC_TYPE_COMP) {
            // MG_Boundary_exchange() that this p_blk depends on has already
            // been performed.  Let's calculate the boundary flux.
            p_blk->flux += MG_Process_boundary_conditions(p_params, p_blk);

            // Apply stencil.
            MG_Stencil(p_params, p_blk);
        }

        p_blk->iter++;
    }
#else
    // Single-threaded.
    MG_Assert(0);
#endif // MG_PARALLEL_TYPE
}
