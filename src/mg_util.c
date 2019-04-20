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

int g_debug_rank = -1;
int g_debug_nprocs = -1;

void MG_Assert_impl(const char *cond, const char *file, int line)
{
    fprintf(stderr, "[pe %d] ** Assert ** `%s` failed (%s : %d).\n",
            g_debug_rank, cond, file, line);
    fflush(stdout);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(-1);
}

void MG_Error_impl(const char *error_msg, const char *file, int line)
{
    fprintf(stderr, "\n\n [pe %d] ** Error ** %s (%s : %d).\n", g_debug_rank,
            error_msg, file, line);
    fflush(stdout);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, -1);
    exit(-1);
}

void MG_Print_header(const Params *p_params)
{
    if (p_params->rank != 0)
        return;

    fprintf(stdout, "\n");
    fprintf(stdout,
            "======================================================= \n");

    switch (p_params->stencil) {
        case (MG_STENCIL_2D5PT):
            fprintf(stdout, " 2D5PT: 5-point stencil in 2 dimensions. \n");
            break;
        case (MG_STENCIL_2D9PT):
            fprintf(stdout, " 2D9PT: 9-point stencil in 2 dimensions. \n");
            break;
        case (MG_STENCIL_3D7PT):
            fprintf(stdout, " 3D7PT: 7-point stencil in 3 dimensions. \n");
            break;
        case (MG_STENCIL_3D27PT):
            fprintf(stdout, " 3D27PT: 27-point stencil in 3 dimensions. \n");
            break;
        default:
            break;
    }
    fprintf(stdout, " Global grid dimension = %d x %d x %d \n",
            p_params->nx * p_params->npx, p_params->ny * p_params->npy,
            p_params->nz * p_params->npz);
    fprintf(stdout, " Local grid dimension  = %d x %d x %d \n", p_params->nx,
            p_params->ny, p_params->nz);
    fprintf(stdout, " Process dimension     = %d x %d x %d \n", p_params->npx,
            p_params->npy, p_params->npz);
    fprintf(stdout, " # of blocks           = %d x %d x %d \n",
            p_params->nblksx, p_params->nblksy, p_params->nblksz);
    fprintf(stdout, " Block cells           = %d x %d x %d \n",
            p_params->ncblkx, p_params->ncblky, p_params->ncblkz);
    fprintf(stdout, " Number of variables   = %d \n", p_params->numvars);
    fprintf(stdout, " %d time steps to be executed. \n", p_params->numtsteps);

    fprintf(stdout,
            "======================================================= \n");
    fprintf(stdout, "\n");
}

void *MG_malloc(size_t size)
{
    void *ptr;
    // Round up.
    size = (size + MG_ALIGNMENT - 1) & ~((size_t)(MG_ALIGNMENT - 1));
    int ret = posix_memalign(&ptr, MG_ALIGNMENT, size);
    MG_Assert(ret == 0);
    return ptr;
}

void *MG_calloc(size_t size1, size_t size2)
{
    void *ptr = MG_malloc(size1 * size2);
    memset(ptr, 0, size1 * size2);
    return ptr;
}
