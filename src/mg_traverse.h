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

#ifndef _mg_traverse_h_
#define _mg_traverse_h_

#include "mg.h"

#define MG_TRAVERSE(_traverse_flag, _p_params, _xstart, _xend, _ystart, _yend, \
                    _zstart, _zend, _func, ...)                                \
    do {                                                                       \
        const Params *p_params = (_p_params);                                  \
        size_t comm_index = 0;                                                 \
        const int xstart = (_xstart);                                          \
        const int xend = (_xend);                                              \
        const int ystart = (_ystart);                                          \
        const int yend = (_yend);                                              \
        const int zstart = (_zstart);                                          \
        const int zend = (_zend);                                              \
        const int traverse_flag = _traverse_flag;                              \
        const int check_diagonal = (p_params->stencil == MG_STENCIL_2D9PT) ||  \
                                   (p_params->stencil == MG_STENCIL_3D27PT);   \
        if (traverse_flag & NEIGHBOR_ANY_ZN) {                                 \
            const int ks = zend;                                               \
            const int ke = ks;                                                 \
            const int ks_halo = zend + 1;                                      \
            const int ke_halo = ks_halo;                                       \
            if (check_diagonal && (traverse_flag & NEIGHBOR_ANY_YN)) {         \
                const int js = yend;                                           \
                const int je = js;                                             \
                const int js_halo = yend + 1;                                  \
                const int je_halo = js_halo;                                   \
                if (traverse_flag & NEIGHBOR_XNYNZN) {                         \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNYNZN, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_X0YNZN) {                         \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0YNZN, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_YNZN) {                           \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_YNZN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            if (check_diagonal && (traverse_flag & NEIGHBOR_ANY_Y0)) {         \
                const int js = ystart;                                         \
                const int je = js;                                             \
                const int js_halo = ystart - 1;                                \
                const int je_halo = js_halo;                                   \
                if (traverse_flag & NEIGHBOR_XNY0ZN) {                         \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNY0ZN, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_X0Y0ZN) {                         \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0Y0ZN, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_Y0ZN) {                           \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Y0ZN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            {                                                                  \
                const int js = ystart;                                         \
                const int je = yend;                                           \
                const int js_halo = ystart;                                    \
                const int je_halo = yend;                                      \
                if (check_diagonal && (traverse_flag & NEIGHBOR_ZNXN)) {       \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_ZNXN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (check_diagonal && (traverse_flag & NEIGHBOR_ZNX0)) {       \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_ZNX0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_ZN) {                             \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_ZN, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        if (traverse_flag & NEIGHBOR_ANY_Z0) {                                 \
            const int ks = zstart;                                             \
            const int ke = ks;                                                 \
            const int ks_halo = zstart - 1;                                    \
            const int ke_halo = ks_halo;                                       \
            if (check_diagonal && (traverse_flag & NEIGHBOR_ANY_YN)) {         \
                const int js = yend;                                           \
                const int je = js;                                             \
                const int js_halo = yend + 1;                                  \
                const int je_halo = js_halo;                                   \
                if (traverse_flag & NEIGHBOR_XNYNZ0) {                         \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNYNZ0, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_X0YNZ0) {                         \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0YNZ0, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_YNZ0) {                           \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_YNZ0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            if (check_diagonal && (traverse_flag & NEIGHBOR_ANY_Y0)) {         \
                const int js = ystart;                                         \
                const int je = js;                                             \
                const int js_halo = ystart - 1;                                \
                const int je_halo = js_halo;                                   \
                if (traverse_flag & NEIGHBOR_XNY0Z0) {                         \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNY0Z0, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_X0Y0Z0) {                         \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0Y0Z0, __VA_ARGS__);                       \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_Y0Z0) {                           \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Y0Z0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            {                                                                  \
                const int js = ystart;                                         \
                const int je = yend;                                           \
                const int js_halo = ystart;                                    \
                const int je_halo = yend;                                      \
                if (check_diagonal && (traverse_flag & NEIGHBOR_Z0XN)) {       \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Z0XN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (check_diagonal && (traverse_flag & NEIGHBOR_Z0X0)) {       \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Z0X0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_Z0) {                             \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Z0, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
        {                                                                      \
            const int ks = zstart;                                             \
            const int ke = zend;                                               \
            const int ks_halo = zstart;                                        \
            const int ke_halo = zend;                                          \
            if (traverse_flag & NEIGHBOR_ANY_YN) {                             \
                const int js = yend;                                           \
                const int je = js;                                             \
                const int js_halo = yend + 1;                                  \
                const int je_halo = js_halo;                                   \
                if (check_diagonal && (traverse_flag & NEIGHBOR_XNYN)) {       \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNYN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (check_diagonal && (traverse_flag & NEIGHBOR_X0YN)) {       \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0YN, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_YN) {                             \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_YN, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            if (traverse_flag & NEIGHBOR_ANY_Y0) {                             \
                const int js = ystart;                                         \
                const int je = js;                                             \
                const int js_halo = ystart - 1;                                \
                const int je_halo = js_halo;                                   \
                if (check_diagonal && (traverse_flag & NEIGHBOR_XNY0)) {       \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XNY0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (check_diagonal && (traverse_flag & NEIGHBOR_X0Y0)) {       \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0Y0, __VA_ARGS__);                         \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_Y0) {                             \
                    const int is = xstart;                                     \
                    const int ie = xend;                                       \
                    const int is_halo = xstart;                                \
                    const int ie_halo = xend;                                  \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_Y0, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
            }                                                                  \
            {                                                                  \
                const int js = ystart;                                         \
                const int je = yend;                                           \
                const int js_halo = ystart;                                    \
                const int je_halo = yend;                                      \
                if (traverse_flag & NEIGHBOR_XN) {                             \
                    const int is = xend;                                       \
                    const int ie = is;                                         \
                    const int is_halo = xend + 1;                              \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_XN, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
                if (traverse_flag & NEIGHBOR_X0) {                             \
                    const int is = xstart;                                     \
                    const int ie = is;                                         \
                    const int is_halo = xstart - 1;                            \
                    const int ie_halo = is_halo;                               \
                    _func(p_params, is, ie, js, je, ks, ke, is_halo, ie_halo,  \
                          js_halo, je_halo, ks_halo, ke_halo, comm_index,      \
                          NEIGHBOR_X0, __VA_ARGS__);                           \
                    comm_index++;                                              \
                }                                                              \
                {                                                              \
                    /* No neighbor. */                                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

#endif /* _mg_traverse_h_ */
