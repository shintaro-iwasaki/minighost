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

static MG_REAL flux_accumulate_2d5pt3d7pt(const Params *p_params,
                                          MG_REAL *grid_vals, BlockInfo *p_blk);

static MG_REAL flux_accumulate_2d9pt3d27pt(const Params *p_params,
                                           MG_REAL *grid_vals,
                                           BlockInfo *p_blk);

// Return a flux value.
MG_REAL MG_Process_boundary_conditions(const Params *p_params, BlockInfo *p_blk)
{
    GridInfo *p_grid = p_blk->p_grid;
    // Stencil iteration
    MG_REAL *grid_vals;
    if (p_blk->iter % 2 == 0) {
        grid_vals = p_grid->values1;
    } else {
        grid_vals = p_grid->values2;
    }
    switch (p_params->stencil) {
        case MG_STENCIL_2D5PT:
        case MG_STENCIL_3D7PT:
            return flux_accumulate_2d5pt3d7pt(p_params, grid_vals, p_blk);
        case MG_STENCIL_2D9PT:
        case MG_STENCIL_3D27PT:
            return flux_accumulate_2d9pt3d27pt(p_params, grid_vals, p_blk);
        default:
            MG_Error("Unknown p_params->stencil.");
    }
    return 0.0;
}

static MG_REAL flux_accumulate_2d5pt3d7pt(const Params *p_params,
                                          MG_REAL *grid_vals, BlockInfo *p_blk)
{
    const int nx = p_params->nx;
    const int ny = p_params->ny;
    const int boundary_flag = p_blk->boundary_flag;
    MG_REAL flux_val = 0.0;

    if (boundary_flag & BOUNDARY_Y0) {
        const int j = 1;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (boundary_flag & BOUNDARY_YN) {
        const int j = p_blk->yend;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (boundary_flag & BOUNDARY_X0) {
        const int i = 1;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (boundary_flag & BOUNDARY_XN) {
        const int i = p_blk->xend;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (boundary_flag & BOUNDARY_Z0) {
        const int k = 1;
        for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (boundary_flag & BOUNDARY_ZN) {
        const int k = p_blk->zend;
        for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }

    if (p_params->stencil == MG_STENCIL_2D5PT) {
        return flux_val * (1.0 / FIVE);
    } else {
        MG_Assert(p_params->stencil == MG_STENCIL_3D7PT);
        return flux_val * (1.0 / SEVEN);
    }
}

static MG_REAL flux_accumulate_2d9pt3d27pt(const Params *p_params,
                                           MG_REAL *grid_vals, BlockInfo *p_blk)
{
    const int nx = p_params->nx;
    const int ny = p_params->ny;

    const int boundary_flag = p_blk->boundary_flag;
    MG_REAL flux_val = 0.0;
    if (boundary_flag & BOUNDARY_Z0) {
        // Z = 0 face
        const int k = 1;
        for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
        // Z = 0 & {X,Y} = {0,N} edges
        if (boundary_flag & BOUNDARY_X0) {
            const int i = 1;
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
            // Z = 0 & X = 0 & Y = {0,N} vertices.
            if (boundary_flag & BOUNDARY_Y0) {
                const int j = 1;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
            if (boundary_flag & BOUNDARY_YN) {
                const int j = ny;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
        }
        if (boundary_flag & BOUNDARY_XN) {
            const int i = nx;
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
            // Z = 0 & X = N & Y = {0,N} vertices.
            if (boundary_flag & BOUNDARY_Y0) {
                const int j = 1;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
            if (boundary_flag & BOUNDARY_YN) {
                const int j = ny;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
        }
        if (boundary_flag & BOUNDARY_Y0) {
            const int j = 1;
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
        if (boundary_flag & BOUNDARY_YN) {
            const int j = ny;
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
    }
    if (boundary_flag & BOUNDARY_ZN) {
        // Z = N face
        const int k = p_params->nz;
        for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
        // Z = N & {X,Y} = {0,N} edges
        if (boundary_flag & BOUNDARY_X0) {
            const int i = 1;
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
            // Z = N & X = 0 & Y = {0,N} vertices.
            if (boundary_flag & BOUNDARY_Y0) {
                const int j = 1;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
            if (boundary_flag & BOUNDARY_YN) {
                const int j = ny;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
        }
        if (boundary_flag & BOUNDARY_XN) {
            const int i = nx;
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
            // Z = N & X = N & Y = {0,N} vertices.
            if (boundary_flag & BOUNDARY_Y0) {
                const int j = 1;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
            if (boundary_flag & BOUNDARY_YN) {
                const int j = ny;
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 1.0;
            }
        }
        if (boundary_flag & BOUNDARY_Y0) {
            const int j = 1;
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
        if (boundary_flag & BOUNDARY_YN) {
            const int j = ny;
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
    }
    if (boundary_flag & BOUNDARY_Y0) {
        // Y = 0 face
        const int j = 1;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
        // Y = 0 & X = {0,N} edges
        if (boundary_flag & BOUNDARY_X0) {
            const int i = 1;
            for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
        if (boundary_flag & BOUNDARY_XN) {
            const int i = nx;
            for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
    }
    if (boundary_flag & BOUNDARY_YN) {
        // Y = N face
        const int j = ny;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int i = p_blk->xstart, ie = p_blk->xend; i <= ie; i++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
        // Y = N & X = {0,N} edges
        if (boundary_flag & BOUNDARY_X0) {
            const int i = 1;
            for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
        if (boundary_flag & BOUNDARY_XN) {
            const int i = nx;
            for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * -3.0;
            }
        }
    }
    if (boundary_flag & BOUNDARY_X0) {
        // X = 0 face
        const int i = 1;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
    }
    if (boundary_flag & BOUNDARY_XN) {
        // X = N face
        const int i = nx;
        for (int k = p_blk->zstart, ke = p_blk->zend; k <= ke; k++) {
            for (int j = p_blk->ystart, je = p_blk->yend; j <= je; j++) {
                flux_val += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)] * 9.0;
            }
        }
    }
    return flux_val * (1.0 / TWENTY_SEVEN);
}
