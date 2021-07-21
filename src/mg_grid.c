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
#include "mg_traverse.h"

static void fill_grid(const Params *p_params, GridInfo *p_grid);
static void block_init(const Params *p_params, GridInfo *p_grid);
static void block_finalize(const Params *p_params, GridInfo *p_grid);
static void debug_grid_init(const Params *p_params, DebugGridInfo *p_debug);
static void debug_grid_finalize(const Params *p_params, DebugGridInfo *p_debug);

void MG_Grid_init(const Params *p_params, GridInfo *p_grid)
{
    MG_Assert(p_grid);

    const int nx = p_params->nx;
    const int ny = p_params->ny;
    const int nz = p_params->nz;

    // Set up cells.
    const size_t num_cells = (nx + 2) * (ny + 2) * (nz + 2);
    p_grid->values1 = (MG_REAL *)MG_calloc(num_cells, sizeof(MG_REAL));
    MG_Assert(p_grid->values1);
    p_grid->values2 = (MG_REAL *)MG_calloc(num_cells, sizeof(MG_REAL));
    MG_Assert(p_grid->values2);
    fill_grid(p_params, p_grid);

    if (p_params->bc_periodic) {
        // Set up periodic boundaries.
        int wraparound_flag = 0;
        MG_Wraparound(p_params, NEIGHBOR_ALL, p_grid->values1, nx, ny, nz, 1,
                      nx, 1, ny, 1, nz);
    }

    // Compute the total source.
    MG_REAL local_source = 0.0;
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= ny; j++) {
            for (int i = 1; i <= nx; i++) {
                local_source +=
                    p_grid->values1[MG_ARRAY_INDEX(i, j, k, nx, ny)];
            }
        }
    }
    p_grid->source_total = 0.0;
    int err = MPI_Allreduce(&local_source, &p_grid->source_total, 1,
                            MG_COMM_REAL, MPI_SUM, MPI_COMM_WORLD);
    MG_Assert(err == MPI_SUCCESS);

    // Initialize variables.
    p_grid->check_answer = 1;
    p_grid->blks = (BlockInfo *)MG_calloc(p_params->nblks, sizeof(BlockInfo));
    MG_Assert(p_grid->blks);
    block_init(p_params, p_grid);

    if (p_params->validation_type & VALIDATE_TYPE_VALUE) {
        p_grid->p_debug = (DebugGridInfo *)MG_calloc(1, sizeof(DebugGridInfo));
        debug_grid_init(p_params, p_grid->p_debug);
    }
}

void MG_Grid_finalize(const Params *p_params, GridInfo *p_grid)
{
    block_finalize(p_params, p_grid);
    free(p_grid->blks);
    free(p_grid->values1);
    free(p_grid->values2);
    p_grid->values1 = NULL;
    p_grid->values2 = NULL;

    if (p_params->validation_type & VALIDATE_TYPE_VALUE) {
        debug_grid_finalize(p_params, p_grid->p_debug);
        free(p_grid->p_debug);
        p_grid->p_debug = NULL;
    }
}

static void fill_grid_impl(const Params *p_params, MG_REAL *values,
                           uint32_t seed)
{
    // Fill input array with random values.
    const int nx = p_params->nx;
    const int ny = p_params->ny;
    const int nz = p_params->nz;

    // Initialize all zones to 0, including ghosts.
    memset(values, 0, sizeof(MG_REAL) * (nx + 2) * (ny + 2) * (nz + 2));

    const uint32_t maxval = MG_Max3i(nx, ny, nz) + 1;
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= ny; j++) {
            for (int i = 1; i <= nx; i++) {
                const MG_REAL val = MG_fast_rand32(&seed) % maxval + 1;
                // For validation, this val must be positive.
                values[MG_ARRAY_INDEX(i, j, k, nx, ny)] = val;
            }
        }
    }
}

static void fill_grid(const Params *p_params, GridInfo *p_grid)
{
    uint32_t seed = p_params->rank * 111 + 77;
    fill_grid_impl(p_params, p_grid->values1, seed);
}

static void debug_grid_init(const Params *p_params, DebugGridInfo *p_debug)
{
    MG_Assert(p_debug);
    const int nx = p_params->nx;
    const int ny = p_params->ny;
    const int nz = p_params->nz;
    const int gnx = p_params->nx * p_params->npx;
    const int gny = p_params->ny * p_params->npy;
    const int gnz = p_params->nz * p_params->npz;
    p_debug->gnx = gnx;
    p_debug->gny = gny;
    p_debug->gnz = gnz;
    const int num_grids = p_params->numtsteps + 1;
    const size_t grid_size =
        (gnx + 2) * (gny + 2) * (gnz + 2) * sizeof(MG_REAL);
    const size_t buffer_size =
        grid_size * num_grids + sizeof(MG_REAL *) * num_grids;
    char *buffer = (char *)MG_calloc(1, buffer_size);
    p_debug->buffer = (void *)buffer;
    // Create an array and set pointers.
    p_debug->p_values = (MG_REAL **)buffer;
    buffer += sizeof(MG_REAL *) * num_grids;
    for (int i = 0; i < num_grids; i++) {
        p_debug->p_values[i] = (MG_REAL *)buffer;
        buffer += grid_size;
    }
    // Initialize the first state.
    MG_REAL *tmp_values =
        (MG_REAL *)MG_calloc(1,
                             (nx + 2) * (ny + 2) * (nz + 2) * sizeof(MG_REAL));
    for (int pz = 0; pz < p_params->npz; pz++) {
        for (int py = 0; py < p_params->npy; py++) {
            for (int px = 0; px < p_params->npx; px++) {
                const int rank =
                    MG_pxpypz_to_world_rank(p_params->npx, p_params->npy,
                                            p_params->npz, px, py, pz);
                uint32_t seed = rank * 111 + 77;
                fill_grid_impl(p_params, tmp_values, seed);
                // Use the seed to copy the value to a global grid.
                for (int k = 1; k <= nz; k++) {
                    const int gk = k + pz * nz;
                    for (int j = 1; j <= ny; j++) {
                        const int gj = j + py * ny;
                        for (int i = 1; i <= nx; i++) {
                            const int gi = i + px * nx;
                            p_debug->p_values[0][MG_ARRAY_INDEX(gi, gj, gk, gnx,
                                                                gny)] =
                                tmp_values[MG_ARRAY_INDEX(i, j, k, nx, ny)];
                        }
                    }
                }
            }
        }
    }
    free(tmp_values);

    if (p_params->bc_periodic) {
        // Set up periodic boundaries.
        MG_Wraparound(p_params, NEIGHBOR_ALL, p_debug->p_values[0], gnx, gny,
                      gnz, 1, gnx, 1, gny, 1, gnz);
    }
    // Compute stencil values in advance.
    for (int step = 0; step < p_params->numtsteps; step++) {
        MG_REAL *grid_in = p_debug->p_values[step];
        MG_REAL *grid_out = p_debug->p_values[step + 1];
        MG_Stencil_kernel(grid_in, grid_out, gnx, gny, gnz, 1, gnx, 1, gny, 1,
                          gnz, p_params->stencil);
        if (p_params->bc_periodic) {
            // Apply periodic boundaries.
            MG_Wraparound(p_params, NEIGHBOR_ALL, p_debug->p_values[step + 1],
                          gnx, gny, gnz, 1, gnx, 1, gny, 1, gnz);
        }
    }
}

static void debug_grid_finalize(const Params *p_params, DebugGridInfo *p_debug)
{
    free(p_debug->buffer);
}

typedef struct {
    int num_comms;
    size_t buffer_lens[26]; // Max 26 (6 faces + 12 lines + 8 points)
    int neighbor_flags[26];
} create_block_count_size_arg_t;

static void create_block_count_size(const Params *p_params_arg,
                                    BlockInfo *p_blk_arg,
                                    create_block_count_size_arg_t *p_func_arg);

static void block_init(const Params *p_params, GridInfo *p_grid)
{
    BlockInfo *blks = p_grid->blks;
    const int px = p_params->px;
    const int py = p_params->py;
    const int pz = p_params->pz;
    const int npx = p_params->npx;
    const int npy = p_params->npy;
    const int npz = p_params->npz;
    const int nblksx = p_params->nblksx;
    const int nblksy = p_params->nblksy;
    const int nblksz = p_params->nblksz;
    //
    // Apply mirroring to assign IDs regarding neighbor processes.
    // This is known to be a good idea to assign VCIs.
    //          7 8 9
    //          4 5 6
    //          1 2 3
    //         -------
    //  3 2 1 | 1 2 3 | 3 2 1
    //  6 5 4 | 4 5 6 | 6 5 4
    //  9 8 7 | 7 8 9 | 9 8 7
    //         -------
    //          7 8 9
    //          4 5 6
    //          1 2 3
    //
    for (int kk = 0; kk < nblksz; kk++) {
        const int k = (pz % 2) ? (nblksz - 1 - kk) : kk;
        const int zstart = k * p_params->ncblkz + 1;
        const int zend = (k + 1) * p_params->ncblkz;
        for (int jj = 0; jj < nblksy; jj++) {
            const int j = (py % 2) ? (nblksy - 1 - jj) : jj;
            const int ystart = j * p_params->ncblky + 1;
            const int yend = (j + 1) * p_params->ncblky;
            for (int ii = 0; ii < nblksx; ii++) {
                const int i = (px % 2) ? (nblksx - 1 - ii) : ii;
                const int xstart = i * p_params->ncblkx + 1;
                const int xend = (i + 1) * p_params->ncblkx;
                const int blkid = ii + nblksx * (jj + nblksy * kk);
                blks[blkid].id = blkid;
                blks[blkid].iter = 0;
                blks[blkid].iter_sync = 0;
                blks[blkid].blkx = i;
                blks[blkid].blky = j;
                blks[blkid].blkz = k;
                int boundary_flag = 0;
                int neighbor_flag = 0;
                int wraparound_flag = 0;
                if (p_params->bc_periodic == 0) {
                    // Set up the boundary flag.
                    int local_boundary_flag = 0;
                    if (i == 0) {
                        if (p_params->px == 0) {
                            boundary_flag |= BOUNDARY_X0;
                        } else {
                            local_boundary_flag |= BOUNDARY_X0;
                        }
                    }
                    if (i == nblksx - 1) {
                        if (p_params->px == p_params->npx - 1) {
                            boundary_flag |= BOUNDARY_XN;
                        } else {
                            local_boundary_flag |= BOUNDARY_XN;
                        }
                    }
                    if (j == 0) {
                        if (p_params->py == 0) {
                            boundary_flag |= BOUNDARY_Y0;
                        } else {
                            local_boundary_flag |= BOUNDARY_Y0;
                        }
                    }
                    if (j == nblksy - 1) {
                        if (p_params->py == p_params->npy - 1) {
                            boundary_flag |= BOUNDARY_YN;
                        } else {
                            local_boundary_flag |= BOUNDARY_YN;
                        }
                    }
                    if (p_params->stencil == MG_STENCIL_3D7PT ||
                        p_params->stencil == MG_STENCIL_3D27PT) {
                        if (k == 0) {
                            if (p_params->pz == 0) {
                                boundary_flag |= BOUNDARY_Z0;
                            } else {
                                local_boundary_flag |= BOUNDARY_Z0;
                            }
                        }
                        if (k == nblksz - 1) {
                            if (p_params->pz == p_params->npz - 1) {
                                boundary_flag |= BOUNDARY_ZN;
                            } else {
                                local_boundary_flag |= BOUNDARY_ZN;
                            }
                        }
                    }
                    neighbor_flag =
                        MG_boundary_to_neighbors(local_boundary_flag,
                                                 p_params->stencil ==
                                                         MG_STENCIL_2D9PT ||
                                                     p_params->stencil ==
                                                         MG_STENCIL_3D27PT);
                } else {
                    // p_params->bc_periodic == 1
                    int local_boundary_flag = 0;
                    int wraparound_boundary_flag = 0;
                    if (i == 0) {
                        local_boundary_flag |= BOUNDARY_X0;
                        if (p_params->npx == 1)
                            wraparound_boundary_flag |= BOUNDARY_X0;
                    }
                    if (i == nblksx - 1) {
                        local_boundary_flag |= BOUNDARY_XN;
                        if (p_params->npx == 1)
                            wraparound_boundary_flag |= BOUNDARY_XN;
                    }
                    if (j == 0) {
                        local_boundary_flag |= BOUNDARY_Y0;
                        if (p_params->npy == 1)
                            wraparound_boundary_flag |= BOUNDARY_Y0;
                    }
                    if (j == nblksy - 1) {
                        local_boundary_flag |= BOUNDARY_YN;
                        if (p_params->npy == 1)
                            wraparound_boundary_flag |= BOUNDARY_YN;
                    }
                    if (p_params->stencil == MG_STENCIL_3D7PT ||
                        p_params->stencil == MG_STENCIL_3D27PT) {
                        if (k == 0) {
                            local_boundary_flag |= BOUNDARY_Z0;
                            if (p_params->npz == 1)
                                wraparound_boundary_flag |= BOUNDARY_Z0;
                        }
                        if (k == nblksz - 1) {
                            local_boundary_flag |= BOUNDARY_ZN;
                            if (p_params->npz == 1)
                                wraparound_boundary_flag |= BOUNDARY_ZN;
                        }
                    }
                    wraparound_flag =
                        MG_boundary_to_neighbors(wraparound_boundary_flag,
                                                 p_params->stencil ==
                                                         MG_STENCIL_2D9PT ||
                                                     p_params->stencil ==
                                                         MG_STENCIL_3D27PT);
                    int neighbor_wraparound_flag =
                        MG_boundary_to_neighbors(local_boundary_flag,
                                                 p_params->stencil ==
                                                         MG_STENCIL_2D9PT ||
                                                     p_params->stencil ==
                                                         MG_STENCIL_3D27PT);
                    neighbor_flag =
                        neighbor_wraparound_flag & (~wraparound_flag);
                }
                blks[blkid].boundary_flag = boundary_flag;
                blks[blkid].neighbor_flag = neighbor_flag;
                blks[blkid].wraparound_flag = wraparound_flag;

                // Set offsets into variables.
                blks[blkid].xstart = xstart;
                blks[blkid].xend = xend;
                blks[blkid].ystart = ystart;
                blks[blkid].yend = yend;
                blks[blkid].zstart = zstart;
                blks[blkid].zend = zend;
                blks[blkid].p_grid = p_grid;
            }
        }
    }
    for (int blkid = 0; blkid < p_params->nblks; blkid++) {
        BlockInfo *p_blk = &blks[blkid];

        // Count CommInfos
        create_block_count_size_arg_t counter;
        counter.num_comms = 0;
        create_block_count_size(p_params, p_blk, &counter);
        // Allocate the whole buffer.
        size_t total_comm_buffer_size = 0;
        for (int i = 0; i < counter.num_comms; i++) {
            total_comm_buffer_size +=
                MG_get_aligned(counter.buffer_lens[i] * sizeof(MG_REAL));
        }
        // It needs a receive buffer, too, so let's double the size.
        const size_t comm_mem_size =
            total_comm_buffer_size * 2 +
            MG_get_aligned(sizeof(CommInfo) * counter.num_comms * 2);

        // Count SyncInfos
        int num_syncs = 0;
        BlockInfo *sync_blks[26]; // Maximum 26 (3D27PT).
        {
            const int adjacents_2d5pt[] = ADJACENTS_2D5PT;
            const int adjacents_2d9pt[] = ADJACENTS_2D9PT;
            const int adjacents_3d7pt[] = ADJACENTS_3D7PT;
            const int adjacents_3d27pt[] = ADJACENTS_3D27PT;
            const int num_adjacents_2d5pt =
                sizeof(adjacents_2d5pt) / sizeof(int);
            const int num_adjacents_2d9pt =
                sizeof(adjacents_2d9pt) / sizeof(int);
            const int num_adjacents_3d7pt =
                sizeof(adjacents_3d7pt) / sizeof(int);
            const int num_adjacents_3d27pt =
                sizeof(adjacents_3d27pt) / sizeof(int);
            const int *adjacents =
                (p_params->stencil == MG_STENCIL_2D5PT)
                    ? adjacents_2d5pt
                    : ((p_params->stencil == MG_STENCIL_2D9PT)
                           ? adjacents_2d9pt
                           : ((p_params->stencil == MG_STENCIL_3D7PT)
                                  ? adjacents_3d7pt
                                  : adjacents_3d27pt));
            const int num_adjacents =
                (p_params->stencil == MG_STENCIL_2D5PT)
                    ? num_adjacents_2d5pt
                    : ((p_params->stencil == MG_STENCIL_2D9PT)
                           ? num_adjacents_2d9pt
                           : ((p_params->stencil == MG_STENCIL_3D7PT)
                                  ? num_adjacents_3d7pt
                                  : num_adjacents_3d27pt));
            for (int adjacent_i = 0; adjacent_i < num_adjacents; adjacent_i++) {
                int adjacent = adjacents[adjacent_i];
                // If adjacent is neither boundary nor neighbor, a local
                // block exists.
                if ((MG_neighbor_to_boundary(adjacent) &
                     p_blk->boundary_flag) ||
                    (MG_neighbor_to_boundary(adjacent) &
                     MG_neighbor_to_boundaries(p_blk->neighbor_flag)))
                    continue;
                // This block is next to a local block (adjacent).
                int x, y, z;
                MG_neighbor_to_sxsysz(adjacent, &x, &y, &z);
                // Because of the mirrored placement, pointer computation is
                // a bit complicated.
                const int adjacent_blkxi =
                    ((px % 2 ? (nblksx - 1 - (p_blk->blkx + x))
                             : (p_blk->blkx + x)) +
                     nblksx) %
                    nblksx;
                const int adjacent_blkyi =
                    ((py % 2 ? (nblksy - 1 - (p_blk->blky + y))
                             : (p_blk->blky + y)) +
                     nblksy) %
                    nblksy;
                const int adjacent_blkzi =
                    ((pz % 2 ? (nblksz - 1 - (p_blk->blkz + z))
                             : (p_blk->blkz + z)) +
                     nblksz) %
                    nblksz;
                BlockInfo *p_adjacent_blk =
                    &blks[adjacent_blkxi +
                          nblksx * (adjacent_blkyi + nblksy * adjacent_blkzi)];
                MG_Assert(p_adjacent_blk->blkx ==
                          (p_blk->blkx + x + nblksx) % nblksx);
                MG_Assert(p_adjacent_blk->blky ==
                          (p_blk->blky + y + nblksy) % nblksy);
                MG_Assert(p_adjacent_blk->blkz ==
                          (p_blk->blkz + z + nblksz) % nblksz);
                sync_blks[num_syncs++] = p_adjacent_blk;
            }
        }
        const size_t sync_mem_size = num_syncs * sizeof(SyncInfo);

        // Allocate memory altogether
        char *buffer = (char *)MG_malloc(comm_mem_size + sync_mem_size);
        p_blk->mem_buffer = buffer;

        // Set up CommInfos
        p_blk->num_comms = counter.num_comms * 2;
        p_blk->comms = (CommInfo *)buffer;
        buffer += MG_get_aligned(sizeof(CommInfo) * counter.num_comms * 2);
        for (int is_unpack = 0; is_unpack <= 1; is_unpack++) {
            for (int i = 0; i < counter.num_comms; i++) {
                CommInfo *p_comm =
                    &p_blk->comms[is_unpack ? (i + counter.num_comms) : i];
                p_comm->buffer = (MG_REAL *)buffer;
                p_comm->buffer_len = counter.buffer_lens[i];
                p_comm->comm = p_params->comms[p_blk->id % p_params->num_comms];
                p_comm->neighbor_flag = counter.neighbor_flags[i];
                int x, y, z;
                MG_neighbor_to_sxsysz(p_comm->neighbor_flag, &x, &y, &z);
                p_comm->rank =
                    MG_pxpypz_to_world_rank(npx, npy, npz, (npx + px + x) % npx,
                                            (npy + py + y) % npy,
                                            (npz + pz + z) % npz);
                // Use different tag for direction.
                const int tag_dir_x = (1 - is_unpack * 2) * x; // -1, 0, 1
                const int tag_dir_y = (1 - is_unpack * 2) * y;
                const int tag_dir_z = (1 - is_unpack * 2) * z;
                p_comm->tag = p_blk->id * 27 +
                              ((tag_dir_z + 1) * 3 + (tag_dir_y + 1)) * 3 +
                              (tag_dir_x + 1);
                buffer +=
                    MG_get_aligned(counter.buffer_lens[i] * sizeof(MG_REAL));
            }
        }

        // Set up SyncInfos
        p_blk->num_syncs = num_syncs;
        p_blk->syncs = (SyncInfo *)buffer;
        buffer += sync_mem_size;
        for (int i = 0; i < num_syncs; i++) {
            p_blk->syncs[i].p_blk = sync_blks[i];
            // p_blk->syncs[i].index will be updated later.
        }
    }
    for (int blkid = 0; blkid < p_params->nblks; blkid++) {
        // Set up SyncInfos->index.
        BlockInfo *p_blk = &blks[blkid];
        for (int i = 0; i < p_blk->num_syncs; i++) {
            BlockInfo *p_adjacent = p_blk->syncs[i].p_blk;
            int index = -1;
            // Find myself.
            for (int j = 0; j < p_adjacent->num_syncs; j++) {
                if (p_adjacent->syncs[j].p_blk == p_blk) {
                    index = j;
                    break;
                }
            }
            MG_Assert(index != -1);
            p_blk->syncs[i].index = index;
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
            MG_Thread_init_sync(&p_blk->syncs[i]);
#endif
        }
    }
}

static void block_finalize(const Params *p_params, GridInfo *p_grid)
{
    BlockInfo *blks = p_grid->blks;
    for (int blkid = 0; blkid < p_params->nblks; blkid++) {
        BlockInfo *p_blk = &blks[blkid];
#if MG_PARALLEL_TYPE & (MG_PARALLEL_TYPE_PTHREADS | MG_PARALLEL_TYPE_ARGOBOTS)
        for (int i = 0; i < p_blk->num_syncs; i++) {
            MG_Thread_init_sync(&p_blk->syncs[i]);
        }
#endif
        free(p_blk->mem_buffer);
    }
}

static void create_block_count_size_f(const Params *p_params, int is, int ie,
                                      int js, int je, int ks, int ke,
                                      int is_halo, int ie_halo, int js_halo,
                                      int je_halo, int ks_halo, int ke_halo,
                                      size_t comm_index, int neighbor_flag,
                                      BlockInfo *p_blk,
                                      create_block_count_size_arg_t *p_arg)
{
    p_arg->buffer_lens[comm_index] =
        (ie - is + 1) * (je - js + 1) * (ke - ks + 1);
    p_arg->neighbor_flags[comm_index] = neighbor_flag;
    p_arg->num_comms++;
}

static void create_block_count_size(const Params *p_params_arg,
                                    BlockInfo *p_blk_arg,
                                    create_block_count_size_arg_t *p_func_arg)
{
    MG_TRAVERSE(p_blk_arg->neighbor_flag, p_params_arg, p_blk_arg->xstart,
                p_blk_arg->xend, p_blk_arg->ystart, p_blk_arg->yend,
                p_blk_arg->zstart, p_blk_arg->zend, create_block_count_size_f,
                p_blk_arg, p_func_arg);
}
