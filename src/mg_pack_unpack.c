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

#define MG_PACK_BUFFER_WRAPPER(p_params, p_blk, is, ie, js, je, ks, ke,        \
                               is_halo, ie_halo, js_halo, je_halo, ks_halo,    \
                               ke_halo, comm_index, neighbor_flag, grid)       \
    do {                                                                       \
        if (ks == ke && js == je) {                                            \
            const int send_comm_index = comm_index;                            \
            const int recv_comm_index = comm_index + p_blk->num_comms / 2;     \
            /* Set the sender buffer pointer. */                               \
            const size_t w = (p_params)->nx;                                   \
            const size_t h = (p_params)->ny;                                   \
            p_blk->comms[send_comm_index].buffer =                             \
                &grid[MG_ARRAY_INDEX(is, js, ks, w, h)];                       \
            /* Set the receive buffer pointer. */                              \
            p_blk->comms[recv_comm_index].buffer =                             \
                &grid[MG_ARRAY_INDEX(is_halo, js_halo, ks_halo, w, h)];        \
        } else {                                                               \
            /* Copy the data from/to buffers. */                               \
            const int send_comm_index = comm_index;                            \
            MG_REAL *__restrict__ buffer_ =                                    \
                (MG_REAL * __restrict__) p_blk->comms[send_comm_index].buffer; \
            const MG_REAL *__restrict__ grid_ = (MG_REAL * __restrict__) grid; \
            size_t offset = 0;                                                 \
            const size_t w = (p_params)->nx;                                   \
            const size_t h = (p_params)->ny;                                   \
            for (int k = (ks), _ke = (ke); k <= _ke; k++) {                    \
                for (int j = (js), _je = (je); j <= _je; j++) {                \
                    for (int i = (is), _ie = (ie); i <= _ie; i++) {            \
                        buffer_[offset++] =                                    \
                            grid_[MG_ARRAY_INDEX(i, j, k, w, h)];              \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

static inline void MG_Pack_buffer_impl(const Params *p_params_arg,
                                       MG_REAL *grid_arg, BlockInfo *p_blk_arg,
                                       int is_unpack_arg)
{
    MG_TRAVERSE_NEIGHBORS(p_params_arg, p_blk_arg, MG_PACK_BUFFER_WRAPPER,
                          grid_arg);
}

void MG_Pack_buffer(const Params *p_params, MG_REAL *grid, BlockInfo *p_blk)
{
    MG_Pack_buffer_impl(p_params, grid, p_blk, 0);
}

#define MG_UNPACK_BUFFER_WRAPPER(p_params, p_blk, is, ie, js, je, ks, ke,      \
                                 is_halo, ie_halo, js_halo, je_halo, ks_halo,  \
                                 ke_halo, comm_index, neighbor_flag, grid)     \
    do {                                                                       \
        if (ks == ke && js == je) {                                            \
            /* Do nothing since the data is directly set. */                   \
        } else {                                                               \
            /* Copy the data from/to buffers. */                               \
            const int recv_comm_index = comm_index + p_blk->num_comms / 2;     \
            const MG_REAL *__restrict__ buffer_ =                              \
                (MG_REAL * __restrict__) p_blk->comms[recv_comm_index].buffer; \
            MG_REAL *__restrict__ grid_ = (MG_REAL * __restrict__) grid;       \
            size_t offset = 0;                                                 \
            const size_t w = (p_params)->nx;                                   \
            const size_t h = (p_params)->ny;                                   \
            for (int k = (ks_halo), _ke = (ke_halo); k <= _ke; k++) {          \
                for (int j = (js_halo), _je = (je_halo); j <= _je; j++) {      \
                    for (int i = (is_halo), _ie = (ie_halo); i <= _ie; i++) {  \
                        grid_[MG_ARRAY_INDEX(i, j, k, w, h)] =                 \
                            buffer_[offset++];                                 \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

static inline void MG_Unpack_buffer_impl(const Params *p_params_arg,
                                         MG_REAL *grid_arg,
                                         BlockInfo *p_blk_arg,
                                         int is_unpack_arg)
{
    MG_TRAVERSE_NEIGHBORS(p_params_arg, p_blk_arg, MG_UNPACK_BUFFER_WRAPPER,
                          grid_arg);
}

void MG_Unpack_buffer(const Params *p_params, MG_REAL *grid, BlockInfo *p_blk)
{
    MG_Unpack_buffer_impl(p_params, grid, p_blk, 1);
}
