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

static void kernel_copy(const MG_REAL *restrict grid_in,
                        MG_REAL *restrict grid_out, int nx, int ny, int nz,
                        int xstart, int xend, int ystart, int yend, int zstart,
                        int zend);
static void kernel_2d5pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend);
static void kernel_2d9pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend);
static void kernel_3d7pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend);
static void kernel_3d27pt(const MG_REAL *restrict grid_in,
                          MG_REAL *restrict grid_out, int nx, int ny, int nz,
                          int xstart, int xend, int ystart, int yend,
                          int zstart, int zend);

void MG_Stencil_kernel(const MG_REAL *restrict grid_in,
                       MG_REAL *restrict grid_out, int nx, int ny, int nz,
                       int xstart, int xend, int ystart, int yend, int zstart,
                       int zend, int stencil_type)
{
    switch (stencil_type) {
        case MG_STENCIL_2D5PT:
            kernel_2d5pt(grid_in, grid_out, nx, ny, nz, xstart, xend, ystart,
                         yend, zstart, zend);
            break;
        case MG_STENCIL_2D9PT:
            kernel_2d9pt(grid_in, grid_out, nx, ny, nz, xstart, xend, ystart,
                         yend, zstart, zend);
            break;
        case MG_STENCIL_3D7PT:
            kernel_3d7pt(grid_in, grid_out, nx, ny, nz, xstart, xend, ystart,
                         yend, zstart, zend);
            break;
        case MG_STENCIL_3D27PT:
            kernel_3d27pt(grid_in, grid_out, nx, ny, nz, xstart, xend, ystart,
                          yend, zstart, zend);
            break;
        default:
            MG_Error("Unknown p_params->stencil");
    }
}

void MG_Stencil(const Params *p_params, BlockInfo *p_blk)
{
    // MG_Stencil_comm_iter() that this p_blk depends on must be performed
    // beforehand.
    GridInfo *p_grid = p_blk->p_grid;
    // Stencil iteration
    MG_REAL *grid_in, *grid_out;
    if (p_blk->iter % 2 == 0) {
        grid_in = p_grid->values1;
        grid_out = p_grid->values2;
    } else {
        grid_in = p_grid->values2;
        grid_out = p_grid->values1;
    }

    // Apply stencil.
    if (p_params->exec_type & MG_EXEC_TYPE_COPYCOMP) {
        kernel_copy(grid_in, grid_out, p_params->nx, p_params->ny, p_params->nz,
                    p_blk->xstart, p_blk->xend, p_blk->ystart, p_blk->yend,
                    p_blk->zstart, p_blk->zend);
    } else {
        MG_Stencil_kernel(grid_in, grid_out, p_params->nx, p_params->ny,
                          p_params->nz, p_blk->xstart, p_blk->xend,
                          p_blk->ystart, p_blk->yend, p_blk->zstart,
                          p_blk->zend, p_params->stencil);
    }
}

static void kernel_copy(const MG_REAL *restrict grid_in,
                        MG_REAL *restrict grid_out, int nx, int ny, int nz,
                        int xstart, int xend, int ystart, int yend, int zstart,
                        int zend)
{
    // REAL_FIXME: loop optimization
#if (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
#pragma omp parallel for
#endif
    for (int k = zstart; k <= zend; k++) {
        for (int j = ystart; j <= yend; j++) {
            for (int i = xstart; i <= xend; i++) {
                grid_out[MG_ARRAY_INDEX(i, j, k, nx, ny)] =
                    grid_in[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }
}

static void kernel_2d5pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend)
{
    // REAL_FIXME: loop optimization
#if (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
#pragma omp parallel for
#endif
    for (int k = zstart; k <= zend; k++) {
        for (int j = ystart; j <= yend; j++) {
            for (int i = xstart; i <= xend; i++) {
                grid_out[MG_ARRAY_INDEX(i, j, k, nx, ny)] =
                    (grid_in[MG_ARRAY_INDEX(i - 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k, nx, ny)]) *
                    (1.0 / FIVE);
            }
        }
    }
}

static void kernel_2d9pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend)
{
#if (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
#pragma omp parallel for
#endif
    for (int k = zstart; k <= zend; k++) {
        for (int j = ystart; j <= yend; j++) {
            for (int i = xstart; i <= xend; i++) {
                grid_out[MG_ARRAY_INDEX(i, j, k, nx, ny)] =
                    (grid_in[MG_ARRAY_INDEX(i - 1, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j + 1, k, nx, ny)]) *
                    (1.0 / NINE);
            }
        }
    }
}

static void kernel_3d7pt(const MG_REAL *restrict grid_in,
                         MG_REAL *restrict grid_out, int nx, int ny, int nz,
                         int xstart, int xend, int ystart, int yend, int zstart,
                         int zend)
{
#if (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
#pragma omp parallel for
#endif
    for (int k = zstart; k <= zend; k++) {
        for (int j = ystart; j <= yend; j++) {
            for (int i = xstart; i <= xend; i++) {
                grid_out[MG_ARRAY_INDEX(i, j, k, nx, ny)] =
                    (grid_in[MG_ARRAY_INDEX(i, j, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k + 1, nx, ny)]) *
                    (1.0 / SEVEN);
            }
        }
    }
}

static void kernel_3d27pt(const MG_REAL *restrict grid_in,
                          MG_REAL *restrict grid_out, int nx, int ny, int nz,
                          int xstart, int xend, int ystart, int yend,
                          int zstart, int zend)
{
#if (MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_OPENMP_FOR)
#pragma omp parallel for
#endif
    for (int k = zstart; k <= zend; k++) {
        for (int j = ystart; j <= yend; j++) {
            for (int i = xstart; i <= xend; i++) {
                grid_out[MG_ARRAY_INDEX(i, j, k, nx, ny)] =
                    (grid_in[MG_ARRAY_INDEX(i - 1, j - 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j + 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j - 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j + 1, k - 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j - 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j + 1, k, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j - 1, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i - 1, j + 1, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j - 1, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i, j + 1, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j - 1, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j, k + 1, nx, ny)] +
                     grid_in[MG_ARRAY_INDEX(i + 1, j + 1, k + 1, nx, ny)]) *
                    (1.0 / TWENTY_SEVEN);
            }
        }
    }
}
