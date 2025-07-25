/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


#ifndef ACADOS_UTILS_TYPES_H_
#define ACADOS_UTILS_TYPES_H_

/* Symbol visibility in DLLs */
#ifndef ACADOS_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define ACADOS_SYMBOL_EXPORT
    #else
      #define ACADOS_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
    #define ACADOS_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define ACADOS_SYMBOL_EXPORT
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

#define MAX_STR_LEN 256
#define ACADOS_EPS 1e-12
#define ACADOS_INFTY 1e10
#define UNUSED(x) ((void)(x))



typedef double real_t;
typedef int int_t;
typedef size_t acados_size_t;


typedef int (*casadi_function_t)(const double** arg, double** res, int* iw, double* w, void* mem);



// enum of return values
enum return_values
{
    ACADOS_SUCCESS = 0,
    ACADOS_NAN_DETECTED = 1,
    ACADOS_MAXITER = 2,
    ACADOS_MINSTEP = 3,
    ACADOS_QP_FAILURE = 4,
    ACADOS_READY = 5,
    ACADOS_UNBOUNDED = 6,
    ACADOS_TIMEOUT = 7,
    ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED = 8,
};


/// Types of the cost function.
typedef enum
{
    LINEAR_LS,
    NONLINEAR_LS,
    CONVEX_OVER_NONLINEAR,
    EXTERNAL,
    INVALID_COST,
} ocp_nlp_cost_t;


/// Types of the cost function.
typedef enum
{
    FIXED_QP_TOL,
    ADAPTIVE_CURRENT_RES_JOINT,
    ADAPTIVE_QPSCALING,
} ocp_nlp_qp_tol_strategy_t;


/// Types of the timeout heuristic.
typedef enum
{
  MAX_CALL,
  MAX_OVERALL,
  LAST,
  AVERAGE,
  ZERO,
} ocp_nlp_timeout_heuristic_t;

// Types of modes for calculating the search direction in SQP_WITH_FEASIBLE_QP
enum search_direction_mode
{
    NOMINAL_QP = 0,
    BYRD_OMOJOKUN = 1,
    FEASIBILITY_QP = 2,
};


/// QP scaling types
typedef enum
{
    NO_OBJECTIVE_SCALING,
    OBJECTIVE_GERSHGORIN,
} qpscaling_scale_objective_type;

/// QP scaling types
typedef enum
{
    NO_CONSTRAINT_SCALING,
    INF_NORM,
} ocp_nlp_qpscaling_constraint_type;



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_TYPES_H_
