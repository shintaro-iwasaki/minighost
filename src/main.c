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

typedef struct {
    int argc;
    char **argv;
} main_kernel_arg_t;

void main_kernel(void *main_arg)
{
    main_kernel_arg_t *p_arg = (main_kernel_arg_t *)main_arg;
    int argc = p_arg->argc;
    char **argv = p_arg->argv;

    Params params;
    MG_Init(argc, argv, &params);
    // Print problem information to stdout.
    if (params.print_header) {
        MG_Print_header(&params);
    }

    // Allocate and initialize grid data.
    // Grid data will create blocks.
    GridInfo *grids = (GridInfo *)MG_calloc(params.numvars, sizeof(GridInfo));
    MG_Assert(grids);
    for (int ivar = 0; ivar < params.numvars; ivar++) {
        GridInfo *p_grid = &grids[ivar];
        MG_Grid_init(&params, p_grid);
    }

    // Main kernel.
    double start_time = 0.0;
    const int warmup_steps = 3;
    for (int tstep = 0; tstep < params.numtsteps;
         tstep += ((tstep == 0) ? warmup_steps : params.check_answer_freq)) {
        const int inc_steps =
            (tstep == 0) ? warmup_steps : params.check_answer_freq;
        const int num_steps = MG_Min(params.numtsteps - tstep, inc_steps);
        if (tstep == warmup_steps)
            start_time = MPI_Wtime();

#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
        // If threaded.
        for (int ivar = 0; ivar < params.numvars; ivar++) {
            GridInfo *p_grid = &grids[ivar];
            for (int blkid = 0, nblks = params.nblks; blkid < nblks; blkid++) {
                p_grid->blks[blkid].arg.p_params = &params;
                p_grid->blks[blkid].arg.p_blk = &p_grid->blks[blkid];
                p_grid->blks[blkid].arg.num_steps = num_steps;
                if (tstep == 0) {
                    MG_Thread_create(MG_Stencil_thread,
                                     &p_grid->blks[blkid].arg,
                                     &p_grid->blks[blkid].thread);
                } else {
                    MG_Thread_revive(&p_grid->blks[blkid].thread);
                }
            }
        }
        for (int ivar = 0; ivar < params.numvars; ivar++) {
            GridInfo *p_grid = &grids[ivar];
            for (int blkid = 0, nblks = params.nblks; blkid < nblks; blkid++) {
                if (tstep + num_steps == params.numtsteps) {
                    MG_Thread_free(&p_grid->blks[blkid].thread);
                } else {
                    MG_Thread_join(&p_grid->blks[blkid].thread);
                }
            }
        }
#else
        // Serialized.
        for (int ivar = 0; ivar < params.numvars; ivar++) {
            GridInfo *p_grid = &grids[ivar];
            // If serialized, all the communication must be done first,
            // especially when the computation needs to access diagonal blocks.
            MG_Stencil_single(&params, p_grid->blks, params.nblks, num_steps);
        }
#endif // MG_PARALLEL_TYPE
       // Intermediate correctness check.
        int cur_steps = tstep + num_steps;
        if (cur_steps != params.numtsteps) {
            for (int ivar = 0; ivar < params.numvars; ivar++) {
                GridInfo *p_grid = &grids[ivar];
                MG_Validate_results(&params, p_grid, cur_steps);
            }
        }
    }
    double end_time = MPI_Wtime();
    // Final correctness check.
    for (int ivar = 0; ivar < params.numvars; ivar++) {
        GridInfo *p_grid = &grids[ivar];
        MG_Validate_results(&params, p_grid, params.numtsteps);
    }
    // Release all the resources
    for (int ivar = 0; ivar < params.numvars; ivar++) {
        GridInfo *p_grid = &grids[ivar];
        MG_Grid_finalize(&params, p_grid);
    }
    free(grids);
    if (params.rank == 0) {
        fprintf(stdout, "Computation & validation finished\n");
        fprintf(stdout, "Elapsed Time: %.4f [ms]\n",
                (end_time - start_time) * 1.0e3 /
                    (params.numtsteps - warmup_steps));
        fprintf(stdout, "Elapsed Time Per Cell: %.4f [ns]\n",
                (end_time - start_time) * 1.0e9 /
                    (params.numtsteps - warmup_steps) /
                    (params.nx * params.npx) / (params.ny * params.npy) /
                    (params.nz * params.npz));
        fflush(stdout);
    }
    MG_Finalize(&params);
}

int main(int argc, char *argv[])
{
    main_kernel_arg_t kernel_arg = { argc, argv };
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
    MG_Thread_session(main_kernel, (void *)&kernel_arg);
#else
    main_kernel((void *)&kernel_arg);
#endif
    return 0;
}
