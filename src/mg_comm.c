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

void MG_Boundary_exchange(const Params *p_params, BlockInfo *p_blk)
{
    // Basically, pack->(send/recv and/or synchronize)->unpack.
    MG_REAL *grid_val = (p_blk->iter % 2 == 0) ? p_blk->p_grid->values1
                                               : p_blk->p_grid->values2;
    // Finish all communication operations.
    if (p_blk->num_comms != 0) {
        MPI_Request requests[26 * 2]; // Max 26 * 2 (send/recv).
        const int num_comms = p_blk->num_comms;
        int err;

        // Pack the data.
        MG_Pack_buffer(p_params, grid_val, p_blk);
        // Start send communication operations.
        for (int i = 0; i < num_comms / 2; i++) {
            CommInfo *p_comm = &p_blk->comms[i];
            err = MPI_Isend(p_comm->buffer, p_comm->buffer_len, MG_COMM_REAL,
                            p_comm->rank, p_comm->tag, p_comm->comm,
                            &requests[i]);
            MG_Assert(err == MPI_SUCCESS);
        }
        // Start receive communication operations.
        for (int i = num_comms / 2; i < num_comms; i++) {
            CommInfo *p_comm = &p_blk->comms[i];
            err = MPI_Irecv(p_comm->buffer, p_comm->buffer_len, MG_COMM_REAL,
                            p_comm->rank, p_comm->tag, p_comm->comm,
                            &requests[i]);
            MG_Assert(err == MPI_SUCCESS);
        }
        err = MPI_Waitall(num_comms, requests, MPI_STATUSES_IGNORE);
        MG_Assert(err == MPI_SUCCESS);
        // Unpack the data.
        MG_Unpack_buffer(p_params, grid_val, p_blk);
    }
}
