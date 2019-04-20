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

static void setup_input_params(int argc, char *argv[], Params *p_params);
static void print_help_message(void);

void MG_Init(int argc, char *argv[], Params *p_params)
{
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
    const int required = MPI_THREAD_MULTIPLE;
#elif (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
    const int required = MPI_THREAD_FUNNELED;
#else
    const int required = MPI_THREAD_SINGLE;
#endif
    int err, provided;
    err = MPI_Init_thread(&argc, &argv, required, &provided);
    MG_Assert(err == MPI_SUCCESS);
    MG_Assert(required == provided);

    // Set up basic MPI things.
    err = MPI_Comm_size(MPI_COMM_WORLD, &p_params->nprocs);
    MG_Assert(err == MPI_SUCCESS);
    g_debug_nprocs = p_params->nprocs;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &p_params->rank);
    MG_Assert(err == MPI_SUCCESS);
    g_debug_rank = p_params->rank;

    setup_input_params(argc, argv, p_params);

    // Check if all processes finish initialization.
    err = MPI_Barrier(MPI_COMM_WORLD);
    MG_Assert(err == MPI_SUCCESS);

    // Set up the rest of the parameters.
    MG_world_rank_to_pxpypz(p_params->rank, p_params->npx, p_params->npy,
                            p_params->npz, &p_params->px, &p_params->py,
                            &p_params->pz);
    p_params->nblks = p_params->nblksx * p_params->nblksy * p_params->nblksz;
    p_params->ncblkx = p_params->nx / p_params->nblksx;
    p_params->ncblky = p_params->ny / p_params->nblksy;
    p_params->ncblkz = p_params->nz / p_params->nblksz;

    // First, create communicators that can talk with neighbor processes.
    p_params->comms = MG_malloc(sizeof(MPI_Comm) * p_params->num_comms);
    MG_Assert(p_params->comms);
    for (int i = 0; i < p_params->num_comms; i++) {
        err = MPI_Comm_dup(MPI_COMM_WORLD, &p_params->comms[i]);
        MG_Assert(err == MPI_SUCCESS);
    }
}

void MG_Finalize(Params *p_params)
{
    int err;
    for (int i = 0; i < p_params->num_comms; i++) {
        err = MPI_Comm_free(&p_params->comms[i]);
        MG_Assert(err == MPI_SUCCESS);
    }
    free(p_params->comms);
    p_params->comms = NULL;
    err = MPI_Finalize();
    MG_Assert(err == MPI_SUCCESS);
}

static void setup_input_params(int argc, char *argv[], Params *p_params)
{
    p_params->nblksx = 1;
    p_params->nblksy = 1;
    p_params->nblksz = 1;

    p_params->nx = 16;
    p_params->ny = 16;
    p_params->nz = 16;

    p_params->npx = 1;
    p_params->npy = 1;
    p_params->npz = 1;

    p_params->num_comms = 1;
    p_params->comm_strategy = 1;

    p_params->stencil = MG_STENCIL_2D5PT;
    p_params->numvars = 1;
    p_params->numtsteps = 10;
    p_params->exec_type = MG_EXEC_TYPE_NORMAL;

    p_params->print_header = 1;
    p_params->error_tol = 1.0e-5;
    p_params->check_answer_freq = 1;
    p_params->validation_type = VALIDATE_TYPE_FLUX;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--nx")) {
            p_params->nx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ny")) {
            p_params->ny = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--nz")) {
            p_params->nz = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--bx")) {
            p_params->nblksx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--by")) {
            p_params->nblksy = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--bz")) {
            p_params->nblksz = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--npx")) {
            p_params->npx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--npy")) {
            p_params->npy = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--npz")) {
            p_params->npz = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--num_comms")) {
            p_params->num_comms = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--numvars")) {
            p_params->numvars = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--numtsteps")) {
            p_params->numtsteps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--stencil")) {
            char stencil_s[128];
            strcpy(stencil_s, argv[++i]);
            if (strcmp(stencil_s, "MG_STENCIL_2D5PT") == 0)
                p_params->stencil = MG_STENCIL_2D5PT;
            else if (strcmp(stencil_s, "MG_STENCIL_2D9PT") == 0)
                p_params->stencil = MG_STENCIL_2D9PT;
            else if (strcmp(stencil_s, "MG_STENCIL_3D7PT") == 0)
                p_params->stencil = MG_STENCIL_3D7PT;
            else if (strcmp(stencil_s, "MG_STENCIL_3D27PT") == 0)
                p_params->stencil = MG_STENCIL_3D27PT;
            else {
                MG_Error("Unknown stencil option");
            }
        } else if (!strcmp(argv[i], "--error_tol")) {
            int error_tol_exp =
                atoi(argv[++i]); // Converted below to real value.
            p_params->error_tol = 1.0 / (MG_REAL)(MG_Pow(10, error_tol_exp));
        } else if (!strcmp(argv[i], "--print_header")) {
            p_params->print_header = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--check_answer_freq")) {
            p_params->check_answer_freq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--validation")) {
            p_params->validation_type = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--exec_type")) {
            p_params->exec_type = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help")) {
            print_help_message();
            MG_Error("Printed help messages");
        } else {
            // Illegal or deprecated parameters.
            MG_Assert(strcmp(argv[i], "--scaling"));
            MG_Assert(strcmp(argv[i], "--blkxlen"));
            MG_Assert(strcmp(argv[i], "--blkylen"));
            MG_Assert(strcmp(argv[i], "--blkzlen"));
            MG_Assert(strcmp(argv[i], "--blkxyzlen"));
            MG_Assert(strcmp(argv[i], "--ndim"));
            MG_Assert(strcmp(argv[i], "--npdim"));
            if (!strcmp(argv[i], "--boundary_condition")) {
                MG_Assert(strcmp(argv[i + 1], "MG_BC_DIRICHLET") == 0);
                i++;
            } else if (!strcmp(argv[i], "--comm_method")) {
                MG_Assert(strcmp(argv[i + 1], "MG_COMM_METHOD_TASK_BLOCKS") ==
                          0);
                i++;
            } else if (!strcmp(argv[i], "--comm_strategy")) {
                MG_Assert(strcmp(argv[i + 1], "MG_COMM_STRATEGY_ISIR") == 0);
                i++;
            } else if (!strcmp(argv[i], "--extra_work_nvars")) {
                int extra_work_nvars = atoi(argv[++i]);
                MG_Assert(extra_work_nvars == 0);
            } else if (!strcmp(argv[i], "--extra_work_percent")) {
                int extra_work_percent = atoi(argv[++i]);
                MG_Assert(extra_work_percent == 0);
            } else if (!strcmp(argv[i], "--debug_grid")) {
                int debug_grid = atoi(argv[++i]);
                MG_Assert(debug_grid == 0);
            } else if (!strcmp(argv[i], "--init_grid_values")) {
                // Only support MG_INIT_GRID_RANDOM.
                MG_Assert(strcmp(argv[i + 1], "MG_INIT_GRID_RANDOM") == 0);
                i++;
            } else if (!strcmp(argv[i], "--block_order")) {
                MG_Assert(strcmp(argv[i + 1], "MG_BLOCK_ORDER_RANDOM") == 0);
                i++;
            } else if (!strcmp(argv[i], "--ghostdepth")) {
                MG_Assert(strcmp(argv[i + 1], "1") == 0);
                i++;
            } else {
                MG_Error("Unknown input parameter");
            }
        }
    }
    if (p_params->check_answer_freq == 0) {
        p_params->check_answer_freq = p_params->numtsteps + 1;
    }
    if (p_params->rank == 0) {
        MG_Assert(p_params->nx > 0);
        MG_Assert(p_params->ny > 0);
        MG_Assert(p_params->nz > 0);
        MG_Assert(p_params->nblksx > 0);
        MG_Assert(p_params->nblksy > 0);
        MG_Assert(p_params->nblksz > 0);
        MG_Assert((p_params->nx % p_params->nblksx) == 0);
        MG_Assert((p_params->ny % p_params->nblksy) == 0);
        MG_Assert((p_params->nz % p_params->nblksz) == 0);
        MG_Assert(p_params->num_comms > 0);
        MG_Assert(p_params->numvars > 0);
        MG_Assert(p_params->numtsteps >= 1);
        MG_Assert(p_params->npx > 0);
        MG_Assert(p_params->npy > 0);
        MG_Assert(p_params->npz > 0);
        MG_Assert(p_params->npx * p_params->npy * p_params->npz ==
                  p_params->nprocs);
        MG_Assert(p_params->check_answer_freq > 0);
        MG_Assert(p_params->validation_type >= 0);
    }
}

static void print_help_message(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "\n (Optional) command line input is of the form: \n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --nx  ( > 0 )\n");
    fprintf(stderr, " --ny  ( > 0 )\n");
    fprintf(stderr, " --nz  ( > 0 )\n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --bx ( > 0 )\n");
    fprintf(stderr, " --by ( > 0 )\n");
    fprintf(stderr, " --bz ( > 0 )\n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --npx ( > 0 )\n");
    fprintf(stderr, " --npy ( > 0 )\n");
    fprintf(stderr, " --npz ( > 0 )\n");
    fprintf(stderr, " npz * npy * npz = number of processes\n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --num_comms ( > 0 )\n");
    fprintf(stderr, " --comm_strategy\n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --stencil \n");
    fprintf(stderr, " --numvars (0 < numvars <= 40)\n");
    fprintf(stderr, " --numtsteps ( > 0 )\n");
    fprintf(stderr, "\n");

    fprintf(stderr, " --error_tol ( e^{-error_tol}; >= 0)\n");
    fprintf(stderr, " --stencil\n");
    fprintf(stderr, " --print_header : 0 or 1 \n");
    fprintf(stderr, " --check_answer_freq : >= 0 \n");
    fprintf(stderr,
            " --validation ( 0: NONE, 0x1: FLUX, 0x2: VALUE, 0x4: PRINT )\n");
    fprintf(stderr, "\n");

    fprintf(stderr,
            " --exec_type ( 1: ONLY_COMM, 2: ONLY_COMP, 4: COPY_COMP )\n");

    MPI_Abort(MPI_COMM_WORLD, -1);
}
