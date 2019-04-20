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
#if MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_PTHREADS

void MG_Thread_session(void (*func)(void *), void *arg)
{
    func(arg);
}

void MG_Thread_self_suspend(SyncInfo *p_sync)
{
    int err;
    err = pthread_mutex_lock(&p_sync->mutex);
    MG_Assert(err == 0);
    while (p_sync->resume_flag == 0) {
        err = pthread_cond_wait(&p_sync->cond, &p_sync->mutex);
        MG_Assert(err == 0);
    }
    p_sync->resume_flag = 0;
    err = pthread_mutex_unlock(&p_sync->mutex);
    MG_Assert(err == 0);
}

void MG_Thread_resume(SyncInfo *p_sync)
{
    int err;
    err = pthread_mutex_lock(&p_sync->mutex);
    MG_Assert(err == 0);
    p_sync->resume_flag = 1;
    err = pthread_cond_signal(&p_sync->cond);
    MG_Assert(err == 0);
    err = pthread_mutex_unlock(&p_sync->mutex);
    MG_Assert(err == 0);
}

void MG_Thread_init_sync(SyncInfo *p_sync)
{
    int err;
    p_sync->resume_flag = 0;
    err = pthread_mutex_init(&p_sync->mutex, NULL);
    MG_Assert(err == 0);
    err = pthread_cond_init(&p_sync->cond, NULL);
    MG_Assert(err == 0);
}

void MG_Thread_finalize_sync(SyncInfo *p_sync)
{
    int err;
    p_sync->resume_flag = 0;
    err = pthread_mutex_destroy(&p_sync->mutex);
    MG_Assert(err == 0);
    err = pthread_cond_destroy(&p_sync->cond);
    MG_Assert(err == 0);
}

static void *pthread_wrapper(void *arg)
{
    MG_thread_t *p_thread = (MG_thread_t *)arg;
    p_thread->func(p_thread->arg);
    return NULL;
}

void MG_Thread_create(void (*func)(void *), void *arg, MG_thread_t *p_thread)
{
    p_thread->func = func;
    p_thread->arg = arg;
    p_thread->terminated = 0;
    int err = pthread_create(&p_thread->thread, NULL, pthread_wrapper,
                             (void *)p_thread);
    MG_Assert(err == 0);
}

void MG_Thread_revive(MG_thread_t *p_thread)
{
    MG_Assert(p_thread->terminated == 1);
    int err = pthread_create(&p_thread->thread, NULL, pthread_wrapper,
                             (void *)p_thread);
    MG_Assert(err == 0);
}

void MG_Thread_join(MG_thread_t *p_thread)
{
    int err = pthread_join(p_thread->thread, NULL);
    MG_Assert(err == 0);
    p_thread->terminated = 1;
}

void MG_Thread_free(MG_thread_t *p_thread)
{
    int err = pthread_join(p_thread->thread, NULL);
    MG_Assert(err == 0);
    p_thread->terminated = 0;
}

#endif // MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_PTHREADS
