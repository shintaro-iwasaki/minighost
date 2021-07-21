.PHONY: buildall buildextra benchmarks

benchmarks: minighost
buildall: minighost
buildextra: minighost

AM_CPPFLAGS = -I$(top_srcdir)/include

bin_PROGRAMS = minighost

minighost_SOURCES = main.c \
					mg_comm.c \
					mg_flux.c \
					mg_grid.c \
					mg_init.c \
					mg_kernel.c \
					mg_pack_unpack.c \
					mg_stencil.c \
					mg_thread_argobots.c \
					mg_thread_pthread.c \
					mg_util.c \
					mg_validate.c

minighost_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) -D_MG_DOUBLE -D_MG_INT8 -DWALL -D_GNU_SOURCE
minighost_CFLAGS = $(AM_CFLAGS) $(CFLAGS) $(MPI_CFLAGS) -std=gnu99
minighost_LDFLAGS = $(MPI_CLDFLAGS)
