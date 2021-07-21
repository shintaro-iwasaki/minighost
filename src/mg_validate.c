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

static void validate_flux(const Params *p_params, const GridInfo *p_grid,
                          int step);
static void validate_value(const Params *p_params, const GridInfo *p_grid,
                           int step);
static void validate_print(const Params *p_params, const GridInfo *p_grid,
                           int step);

// Only one thread can call this.  This is collective.
void MG_Validate_results(const Params *p_params, const GridInfo *p_grid,
                         int step)
{
    if (p_params->validation_type & VALIDATE_TYPE_PRINT) {
        validate_print(p_params, p_grid, step);
    }
    if (p_params->validation_type & VALIDATE_TYPE_FLUX) {
        validate_flux(p_params, p_grid, step);
    }
    if (p_params->validation_type & VALIDATE_TYPE_VALUE) {
        validate_value(p_params, p_grid, step);
    }
}

static inline MG_REAL get_err(MG_REAL val1, MG_REAL val2)
{
    MG_REAL diff;
    if (-1.0e-8 < val2 && val2 < 1.0e-8) {
        // Too small.
        diff = (val1 - val2) * 1.0e8;
    } else {
        diff = (val1 - val2) / val2;
    }
    return diff < 0.0 ? (-diff) : diff;
}

static void validate_flux(const Params *p_params, const GridInfo *p_grid,
                          int step)
{
    // So basically SUM(flux) + SUM(grid_vals) == source_total.
    // Sum up fluxes of local blocks.
    MG_REAL local_flux = 0.0;
    for (int i = 0; i < p_params->nblks; i++) {
        local_flux += p_grid->blks[i].flux;
    }
    // Sum up all local cell values.
    MG_REAL local_sum = 0.0;
    for (int ivar = 0; ivar < p_params->numvars; ivar++) {
        MG_REAL *grid_vals;
        if (step % 2) {
            grid_vals = p_grid->values2;
        } else {
            grid_vals = p_grid->values1;
        }
        // Local grid summed.
        const int nx = p_params->nx;
        const int ny = p_params->ny;
        const int nz = p_params->nz;
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= ny; j++) {
                for (int i = 1; i <= nx; i++) {
                    local_sum += grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
                }
            }
        }
    }
    // Compare those values.  Let's use reduction.
    MG_REAL in_vals[2] = { local_flux, local_sum };
    MG_REAL out_vals[2] = { 0.0, 0.0 };
    int err = MPI_Allreduce(in_vals, out_vals, 2, MG_COMM_REAL, MPI_SUM,
                            MPI_COMM_WORLD);
    MG_Assert(err == MPI_SUCCESS);
    // Check the value.
    const MG_REAL err_val =
        get_err(out_vals[0] + out_vals[1], p_grid->source_total);
    if (err_val > p_params->error_tol) {
        if (p_params->rank == 0) {
            fprintf(stderr,
                    "  ** Validation error: err is %8.8e, (tolerance of %e.)\n"
                    "     (src = %8.8e, flux = %8.8e, sum = %8.8e)\n",
                    err_val, p_params->error_tol, p_grid->source_total,
                    out_vals[0], out_vals[1]);
            fflush(stderr);
            MG_Error("Validation failed.  Abort.\n");
        }
    }
}

static void validate_value(const Params *p_params, const GridInfo *p_grid,
                           int step)
{
    int err, num_errs = 0;
    for (int rank = 0; rank < p_params->nprocs; rank++) {
        err = MPI_Barrier(MPI_COMM_WORLD);
        MG_Assert(err == MPI_SUCCESS);
        if (rank != p_params->rank) {
            continue;
        }
        MG_REAL *grid_vals;
        if (step % 2) {
            grid_vals = p_grid->values2;
        } else {
            grid_vals = p_grid->values1;
        }
        MG_Assert(p_grid->p_debug);
        MG_Assert(step <= p_params->numtsteps);
        const MG_REAL *ans_values = p_grid->p_debug->p_values[step];
        // Compare values.
        const int nx = p_params->nx;
        const int ny = p_params->ny;
        const int nz = p_params->nz;
        const int gnx = p_grid->p_debug->gnx;
        const int gny = p_grid->p_debug->gny;
        int px, py, pz;
        MG_world_rank_to_pxpypz(p_params->rank, p_params->npx, p_params->npy,
                                p_params->npz, &px, &py, &pz);
        for (int k = 1; k <= nz; k++) {
            const int gk = k + pz * nz;
            for (int j = 1; j <= ny; j++) {
                const int gj = j + py * ny;
                for (int i = 1; i <= nx; i++) {
                    const int gi = i + px * nx;
                    const MG_REAL ans =
                        p_grid->p_debug
                            ->p_values[step]
                                      [MG_ARRAY_INDEX(gi, gj, gk, gnx, gny)];
                    const MG_REAL ret =
                        grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
                    MG_REAL err_val = get_err(ans, ret);
                    if (err_val > p_params->error_tol) {
                        fprintf(stderr,
                                "  ** Validation error: err is %8.8e, "
                                "(tolerance of %e.)\n"
                                "     (rank = %d value[%d][%d][%d] (=%f) != "
                                "ans (=%f))\n",
                                err_val, p_params->error_tol, p_params->rank, i,
                                j, k, ret, ans);
                        num_errs++;
                        if (num_errs >= 5) {
                            fprintf(stderr, "     Perhaps more ...\n");
                            goto EXIT_FOR;
                        }
                    }
                }
            }
        }
    EXIT_FOR:
        fflush(stderr);
    }

    int num_total_errs = 0;
    err = MPI_Allreduce(&num_errs, &num_total_errs, 1, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD);
    MG_Assert(err == MPI_SUCCESS);
    if (num_total_errs != 0 && p_params->rank == 0) {
        MG_Error("Validation failed.  Abort.\n");
    }
}

static void validate_print(const Params *p_params, const GridInfo *p_grid,
                           int step)
{
    int err, num_errs = 0;
    for (int rank = 0; rank < p_params->nprocs; rank++) {
        err = MPI_Barrier(MPI_COMM_WORLD);
        MG_Assert(err == MPI_SUCCESS);
        if (rank != p_params->rank) {
            continue;
        }
        for (int is_new = 0; is_new <= 1; is_new++) {
            MG_REAL *grid_vals;
            if ((step + is_new) % 2) {
                grid_vals = p_grid->values1;
            } else {
                grid_vals = p_grid->values2;
            }
            // Compare values.
            const int nx = p_params->nx;
            const int ny = p_params->ny;
            const int nz = p_params->nz;

            fprintf(stdout, "[%d | %s] rank = %d\n", step,
                    (is_new ? "new" : "old"), rank);
            int k_from = (p_params->stencil == MG_STENCIL_2D5PT ||
                          p_params->stencil == MG_STENCIL_2D9PT || is_new)
                             ? 1
                             : 0;
            int k_to = (p_params->stencil == MG_STENCIL_2D5PT ||
                        p_params->stencil == MG_STENCIL_2D9PT || is_new)
                           ? nz
                           : (nz + 1);
            for (int k = k_from; k <= k_to; k++) {
                if (nz != 1) {
                    fprintf(stdout, "  k = %d\n", k);
                }
                int j_from = is_new ? 1 : 0;
                int j_to = is_new ? ny : (ny + 1);
                for (int j = j_from; j <= j_to; j++) {
                    fprintf(stdout, "   ");
                    int i_from = is_new ? 1 : 0;
                    int i_to = is_new ? nx : (nx + 1);
                    for (int i = i_from; i <= i_to; i++) {
                        const MG_REAL val =
                            grid_vals[MG_ARRAY_INDEX(i, j, k, nx, ny)];
                        // It looks like:
                        // "-1.9e+07"
                        // " 7.2e+00"
                        fprintf(stdout, " % 1.2e", val);
                    }
                    fprintf(stdout, "\n");
                }
            }
            fprintf(stdout, "\n");
            fflush(NULL);
        }
    }
    err = MPI_Barrier(MPI_COMM_WORLD);
    MG_Assert(err == MPI_SUCCESS);
}
