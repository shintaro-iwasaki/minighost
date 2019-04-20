
AC_INIT([minighost], [.01], [siwasaki@anl.gov])
AM_INIT_AUTOMAKE([-Wall foreign]) # -Werror
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_FC
CFLAGS="$CFLAGS -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-parameter -Wno-unused-but-set-variable"

AC_ARG_ENABLE([debug],
              [AS_HELP_STRING([--enable-debug],[Enable Minighost debug options])])

AS_IF([test "x$enable_debug" = "xyes"],
      [AC_DEFINE([_MG_DEBUG], [1], [Defined if debug support is enabled])])

AC_ARG_WITH([runtime],
            [AS_HELP_STRING([--with-runtime],
                            [choose which runtime to use (serial, openmp, pthreads, argobots)])])

AS_IF([test "x$with_runtime" != "x"],
    [case "$with_runtime" in
        serial)
            AC_DEFINE([_MG_SERIAL], [1], [Defined to execute serially])
            ;;
        openmp)
            AC_DEFINE([_MG_OPENMP], [1], [Defined to use OpenMP runtime])
            ;;
        pthreads)
            ;;
        argobots)
            AC_DEFINE([_MG_ARGOBOTS], [1], [Defined to use Argobots runtime])
            AC_CHECK_HEADER([abt.h],[],AC_MSG_ERROR([unable to find Argobots header file]))
            AC_CHECK_LIB([abt],[ABT_init],[],AC_MSG_ERROR([unable to find Argobots library]))
            ;;
        *)
            AC_MSG_ERROR([Unknown runtime])
            ;;
        esac])

AC_CHECK_LIB([pthread],[pthread_create],[],AC_MSG_ERROR([unable to find Pthreads library]))

AC_CONFIG_HEADERS([config.h])
    AC_CONFIG_FILES([
            Makefile
            src/Makefile
    ])

#ref/Makefile
AC_OUTPUT
