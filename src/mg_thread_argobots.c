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
#if MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_ARGOBOTS

typedef struct abt_global_t {
    char pad1[64];
    int num_xstreams;
    int num_vcis;
#ifdef MG_ADVANCED_SCHED
    int *num_assoc_threads;
#endif
    ABT_xstream *xstreams;
    ABT_pool *shared_pools, *priv_pools;
    ABT_sched *scheds;
    ABT_xstream_barrier xstream_barrier;
    int first_init;
    char pad2[64];
} abt_global_t;
abt_global_t g_abt_global;

#define MG_NUM_ASSOC_THREAD_THRESHOLD 64

static inline uint32_t xorshift_rand32(uint32_t *p_seed)
{
    /* George Marsaglia, "Xorshift RNGs", Journal of Statistical Software,
     * Articles, 2003 */
    uint32_t seed = *p_seed;
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    *p_seed = seed;
    return seed;
}

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    return ABT_SUCCESS;
}

static void sched_run(ABT_sched sched)
{
    const int work_count_mask_local = 16 - 1;
    const int work_count_mask_remote = 256 - 1;
    const int work_count_mask_event = 8192 - 1;
    int rank;
    ABT_self_get_xstream_rank(&rank);

    // Set up affinity.
    ABT_xstream self_xstream;
    ABT_self_get_xstream(&self_xstream);
    ABT_xstream_set_cpubind(self_xstream, rank);

    int my_vcimask = 0;
    if (g_abt_global.first_init == 0) {
        for (int i = rank; i < g_abt_global.num_vcis;
             i += g_abt_global.num_xstreams) {
            my_vcimask += 1 << (i + 1);
        }
        if (rank == 0) {
            // Someone needs to take care of vci[0]
            MPIX_Set_exp_info(MPIX_INFO_TYPE_VCIMASK, NULL, my_vcimask + 1);
        } else {
            MPIX_Set_exp_info(MPIX_INFO_TYPE_VCIMASK, NULL, my_vcimask);
        }
        ABT_xstream_barrier_wait(g_abt_global.xstream_barrier);
        g_abt_global.first_init = 1;
    }
    int num_pools;
    ABT_sched_get_num_pools(sched, &num_pools);
    ABT_pool *all_pools = (ABT_pool *)malloc(num_pools * sizeof(ABT_pool));
    ABT_sched_get_pools(sched, num_pools, 0, all_pools);
    ABT_pool my_shared_pool = all_pools[0];
    ABT_pool my_priv_pool = all_pools[1];
    int num_shared_pools = num_pools - 2;
    ABT_pool *shared_pools = all_pools + 2;

    uint32_t seed = (uint32_t)((intptr_t)all_pools);

    int work_count = 0;
    while (1) {
        int local_work_count = 0;
        ABT_unit unit;
        /* Try to pop a ULT from a local pool */
        ABT_pool_pop(my_priv_pool, &unit);
        if (unit != ABT_UNIT_NULL) {
#ifdef MG_ADVANCED_SCHED
            g_abt_global.num_assoc_threads[rank] =
                MG_NUM_ASSOC_THREAD_THRESHOLD;
#endif
            /* Move this unit to my_shared_pool. */
            ABT_xstream_run_unit(unit, my_priv_pool);
            local_work_count++;
            work_count++;
        }
        if (local_work_count == 0 ||
            ((work_count & work_count_mask_local) == 0)) {
            ABT_pool_pop(my_shared_pool, &unit);
            if (unit != ABT_UNIT_NULL) {
                ABT_xstream_run_unit(unit, my_shared_pool);
                local_work_count++;
                work_count++;
            }
        }
        if (local_work_count == 0 ||
            ((work_count & work_count_mask_remote) == 0)) {
            /* RWS */
            if (num_shared_pools > 0) {
                uint32_t rand_num = xorshift_rand32(&seed);
                ABT_pool victim_pool =
                    shared_pools[rand_num % num_shared_pools];
                ABT_pool_pop(victim_pool, &unit);
                if (unit != ABT_UNIT_NULL) {
                    ABT_unit_set_associated_pool(unit, my_shared_pool);
                    ABT_xstream_run_unit(unit, my_shared_pool);
                    local_work_count++;
                    work_count++;
                }
            }
        }
        work_count++;
        if ((work_count & work_count_mask_event) == 0) {
            ABT_bool stop;
            ABT_xstream_check_events(sched);
            ABT_sched_has_to_stop(sched, &stop);
            if (stop == ABT_TRUE) {
                break;
            }
        }
    }
    free(all_pools);
}

static int sched_free(ABT_sched sched)
{
    return ABT_SUCCESS;
}

static void thread_init(void)
{
    int ret;
    ret = ABT_init(0, 0);
    assert(ret == ABT_SUCCESS);

    int num_xstreams = 1;
    if (getenv("ABT_NUM_XSTREAMS")) {
        num_xstreams = atoi(getenv("ABT_NUM_XSTREAMS"));
        if (num_xstreams < 0)
            num_xstreams = 1;
    }
    int num_vcis = 1;
    if (getenv("MPIR_CVAR_CH4_NUM_VCIS")) {
        num_vcis = atoi(getenv("MPIR_CVAR_CH4_NUM_VCIS"));
    }

    g_abt_global.num_xstreams = num_xstreams;
    g_abt_global.num_vcis = num_vcis;
    g_abt_global.xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    g_abt_global.shared_pools =
        (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    g_abt_global.priv_pools =
        (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    g_abt_global.scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);
#ifdef MG_ADVANCED_SCHED
    g_abt_global.num_assoc_threads =
        (int *)calloc(sizeof(int), num_xstreams + 16);
#endif
    ret =
        ABT_xstream_barrier_create(num_xstreams, &g_abt_global.xstream_barrier);
    assert(ret == ABT_SUCCESS);
    /* Create pools. */
    for (int i = 0; i < num_xstreams; i++) {
        ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                                    ABT_TRUE, &g_abt_global.shared_pools[i]);
        assert(ret == ABT_SUCCESS);
        ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                                    ABT_TRUE, &g_abt_global.priv_pools[i]);
        assert(ret == ABT_SUCCESS);
    }
    /* Create schedulers. */
    ABT_sched_def sched_def = {
        .type = ABT_SCHED_TYPE_ULT,
        .init = sched_init,
        .run = sched_run,
        .free = sched_free,
        .get_migr_pool = NULL,
    };
    for (int i = 0; i < num_xstreams; i++) {
        ABT_pool *tmp = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams + 1);
        int pool_index = 0;
        tmp[pool_index++] = g_abt_global.shared_pools[i];
        tmp[pool_index++] = g_abt_global.priv_pools[i];
        for (int j = 1; j < num_xstreams; j++) {
            tmp[pool_index++] =
                g_abt_global.shared_pools[(i + j) % num_xstreams];
        }
        ret = ABT_sched_create(&sched_def, num_xstreams + 1, tmp,
                               ABT_SCHED_CONFIG_NULL, &g_abt_global.scheds[i]);
        assert(ret == ABT_SUCCESS);
        free(tmp);
    }

    /* Create secondary execution streams. */
    for (int i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(g_abt_global.scheds[i],
                                 &g_abt_global.xstreams[i]);
        assert(ret == ABT_SUCCESS);
    }

    /* Set up a primary execution stream. */
    ret = ABT_xstream_self(&g_abt_global.xstreams[0]);
    assert(ret == ABT_SUCCESS);
    ret = ABT_xstream_set_main_sched(g_abt_global.xstreams[0],
                                     g_abt_global.scheds[0]);
    assert(ret == ABT_SUCCESS);

    /* Execute a scheduler once. */
    ret = ABT_self_yield();
    assert(ret == ABT_SUCCESS);
}

static inline void thread_finalize(void)
{
    int ret;
    /* Join secondary execution streams. */
    for (int i = 1; i < g_abt_global.num_xstreams; i++) {
        ret = ABT_xstream_join(g_abt_global.xstreams[i]);
        assert(ret == ABT_SUCCESS);
        ret = ABT_xstream_free(&g_abt_global.xstreams[i]);
        assert(ret == ABT_SUCCESS);
    }
    /* Free secondary execution streams' schedulers */
    for (int i = 1; i < g_abt_global.num_xstreams; i++) {
        ret = ABT_sched_free(&g_abt_global.scheds[i]);
        assert(ret == ABT_SUCCESS);
    }
    ret = ABT_xstream_barrier_free(&g_abt_global.xstream_barrier);
    assert(ret == ABT_SUCCESS);

    ret = ABT_finalize();
    assert(ret == ABT_SUCCESS);
    free(g_abt_global.xstreams);
    g_abt_global.xstreams = NULL;
    free(g_abt_global.shared_pools);
    g_abt_global.shared_pools = NULL;
    free(g_abt_global.priv_pools);
    g_abt_global.priv_pools = NULL;
    free(g_abt_global.scheds);
    g_abt_global.scheds = NULL;
#ifdef MG_ADVANCED_SCHED
    free(g_abt_global.num_assoc_threads);
    g_abt_global.num_assoc_threads = NULL;
#endif
}

void MG_Thread_session(void (*func)(void *), void *arg)
{
    int err;

    thread_init();

    ABT_thread main_thread;
    ABT_pool pool = g_abt_global.priv_pools[0];
    err =
        ABT_thread_create(pool, func, arg, ABT_THREAD_ATTR_NULL, &main_thread);
    MG_Assert(err == ABT_SUCCESS);
    err = ABT_thread_free(&main_thread);
    MG_Assert(err == ABT_SUCCESS);

    thread_finalize();
}

void MG_Thread_self_suspend(SyncInfo *p_sync)
{
    int err;

    ABT_thread self_thread;
    err = ABT_self_get_thread(&self_thread);
    MG_Assert(err == ABT_SUCCESS);
    // If this is the first time, update the value.  To avoid polluting
    // caches, we add a branch.
    if (__atomic_load_n(&p_sync->thread, __ATOMIC_ACQUIRE) != self_thread) {
        // Only this thread accesses it (except for initialization), so relaxed
        // is fine.
        __atomic_store_n(&p_sync->thread, self_thread, __ATOMIC_RELEASE);
    }
    // Sleep.
#ifdef MG_ADVANCED_SCHED
    // If this is the only thread, let's put this thread to the private pool.
    int rank;
    err = ABT_self_get_xstream_rank(&rank);
    MG_Assert(err == ABT_SUCCESS);
    if (g_abt_global.num_assoc_threads[rank] < MG_NUM_ASSOC_THREAD_THRESHOLD) {
        // Maybe this thread is suitable for that private pool.
        if (++g_abt_global.num_assoc_threads[rank] ==
            MG_NUM_ASSOC_THREAD_THRESHOLD) {
            ABT_thread_set_associated_pool(self_thread,
                                           g_abt_global.priv_pools[rank]);
        }
    }
#endif
    err = ABT_self_suspend();
    MG_Assert(err == ABT_SUCCESS);
}

void MG_Thread_resume(SyncInfo *p_sync)
{
    int err;
    ABT_thread thread;
    do {
        thread = __atomic_load_n(&p_sync->thread, __ATOMIC_ACQUIRE);
    } while (thread == ABT_THREAD_NULL);
    // Wait until it gets blocked.
    ABT_thread_state state;
    do {
        err = ABT_thread_get_state(thread, &state);
        MG_Assert(err == ABT_SUCCESS);
    } while (state != ABT_THREAD_STATE_BLOCKED);
    // Let's resume that thread.
    err = ABT_thread_resume(thread);
    MG_Assert(err == ABT_SUCCESS);
}

void MG_Thread_init_sync(SyncInfo *p_sync)
{
    p_sync->thread = ABT_THREAD_NULL;
}

void MG_Thread_finalize_sync(SyncInfo *p_sync)
{
    // Do nothing.
}

static void argobots_wrapper(void *arg)
{
    int err;
    MG_thread_t *p_thread = (MG_thread_t *)arg;
    while (__atomic_load_n(&p_thread->terminated, __ATOMIC_ACQUIRE) == 0) {
        p_thread->func(p_thread->arg);
        // Finish the kernel.  Updated the "terminated" flag.
        while (1) {
            // It failed.  Maybe someone is sleeping on this thread.
            if (__atomic_load_n(&p_thread->terminated, __ATOMIC_ACQUIRE) == 2) {
                // Wake up that thread.
                __atomic_store_n(&p_thread->waiter, 1, __ATOMIC_RELAXED);
                ABT_thread_state state;
                do {
                    err = ABT_thread_get_state(p_thread->waiter, &state);
                    MG_Assert(err == ABT_SUCCESS);
                } while (state != ABT_THREAD_STATE_BLOCKED);
                err = ABT_thread_resume(p_thread->waiter);
                MG_Assert(err == ABT_SUCCESS);
                break;
            }
            int expected = 0;
            if (__atomic_compare_exchange_n(&p_thread->terminated, &expected, 1,
                                            1, __ATOMIC_ACQ_REL,
                                            __ATOMIC_ACQUIRE)) {
                break;
            }
        }
        // Let's wait for the next resume.
        err = ABT_self_suspend();
        MG_Assert(err == ABT_SUCCESS);
    }
    // This thread is asked to terminate.
}

void MG_Thread_create(void (*func)(void *), void *arg, MG_thread_t *p_thread)
{
    p_thread->func = func;
    p_thread->arg = arg;
    p_thread->terminated = 0;
    int ret, rank;
    ret = ABT_self_get_xstream_rank(&rank);
    MG_Assert(ret == ABT_SUCCESS);
    ABT_pool pool = g_abt_global.shared_pools[rank];
    ret = ABT_thread_create(pool, argobots_wrapper, p_thread,
                            ABT_THREAD_ATTR_NULL, &p_thread->thread);
    MG_Assert(ret == ABT_SUCCESS);
}

void MG_Thread_revive(MG_thread_t *p_thread)
{
    int err;
    ABT_thread_state state;
    do {
        err = ABT_thread_get_state(p_thread->thread, &state);
        MG_Assert(err == ABT_SUCCESS);
    } while (state != ABT_THREAD_STATE_BLOCKED);
    __atomic_store_n(&p_thread->terminated, 0, __ATOMIC_RELAXED);
    err = ABT_thread_resume(p_thread->thread);
    MG_Assert(err == ABT_SUCCESS);
}

static inline void thread_join(MG_thread_t *p_thread)
{
    while (__atomic_load_n(&p_thread->terminated, __ATOMIC_ACQUIRE) == 0) {
        int err, expected = 0;
        ABT_thread self_thread;
        err = ABT_self_get_thread(&self_thread);
        MG_Assert(err == ABT_SUCCESS);
        p_thread->waiter = self_thread;
        if (__atomic_compare_exchange_n(&p_thread->terminated, &expected, 2, 1,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            // terminated becomes 2.  Let's wait.
            err = ABT_self_suspend();
            MG_Assert(err == ABT_SUCCESS);
        }
    }
    // Woken up by p_thread.
}

void MG_Thread_join(MG_thread_t *p_thread)
{
    thread_join(p_thread);
}

void MG_Thread_free(MG_thread_t *p_thread)
{
    int err;
    thread_join(p_thread);
    // Resume the thread without changing terminated to 0.
    ABT_thread_state state;
    do {
        err = ABT_thread_get_state(p_thread->thread, &state);
        MG_Assert(err == ABT_SUCCESS);
    } while (state != ABT_THREAD_STATE_BLOCKED);
    err = ABT_thread_resume(p_thread->thread);
    MG_Assert(err == ABT_SUCCESS);
    err = ABT_thread_free(&p_thread->thread);
    MG_Assert(err == ABT_SUCCESS);
}

#endif // MG_PARALLEL_TYPE & MG_PARALLEL_TYPE_ARGOBOTS
