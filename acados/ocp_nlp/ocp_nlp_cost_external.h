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


#ifndef ACADOS_OCP_NLP_OCP_NLP_COST_EXTERNAL_H_
#define ACADOS_OCP_NLP_OCP_NLP_COST_EXTERNAL_H_

#ifdef __cplusplus
extern "C" {
#endif

// blasfeo
#include "blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"

/************************************************
 * dims
 ************************************************/

typedef struct
{
    int nx;  // number of states
    int nz;  // number of algebraic variables
    int nu;  // number of inputs
    int ns;  // number of slacks
    int np; // number of parameters
    int np_global; // number of global parameters
} ocp_nlp_cost_external_dims;

//
acados_size_t ocp_nlp_cost_external_dims_calculate_size(void *config);
//
void *ocp_nlp_cost_external_dims_assign(void *config, void *raw_memory);
//
void ocp_nlp_cost_external_dims_set(void *config_, void *dims_, const char *field, int* value);
//
void ocp_nlp_cost_external_dims_get(void *config_, void *dims_, const char *field, int* value);

/************************************************
 * model
 ************************************************/

typedef struct
{
    external_function_generic *ext_cost_fun;  // function
    external_function_generic *ext_cost_fun_jac_hess;  // function, gradient and hessian
    external_function_generic *ext_cost_fun_jac;  // function, gradient
    external_function_generic *ext_cost_hess_xu_p;  // jacobian of cost gradient wrt params
    external_function_generic *ext_cost_grad_p; // gradient of the cost wrt paraams
    struct blasfeo_dvec Z;
    struct blasfeo_dvec z;
    struct blasfeo_dmat numerical_hessian;  // custom hessian approximation
    double scaling;
} ocp_nlp_cost_external_model;

//
acados_size_t ocp_nlp_cost_external_model_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_external_model_assign(void *config, void *dims, void *raw_memory);



/************************************************
 * options
 ************************************************/

typedef struct
{
    int use_numerical_hessian;  // > 0 indicating custom hessian is used instead of CasADi evaluation
    int with_solution_sens_wrt_params;
    int add_hess_contribution;
} ocp_nlp_cost_external_opts;

//
acados_size_t ocp_nlp_cost_external_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_external_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_cost_external_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_external_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_external_opts_set(void *config, void *opts, const char *field, void *value);


/************************************************
 * memory
 ************************************************/

typedef struct
{
    struct blasfeo_dmat *jac_lag_stat_p_global;    // pointer to jacobian of stationarity condition wrt parameters
    struct blasfeo_dvec grad;    // gradient of cost function
    struct blasfeo_dvec *ux;     // pointer to ux in nlp_out
    struct blasfeo_dmat *RSQrq;  // pointer to RSQrq in qp_in
    struct blasfeo_dvec *Z;      // pointer to Z in qp_in
    struct blasfeo_dvec *z_alg;         ///< pointer to z in sim_out
    struct blasfeo_dmat *dzdux_tran;    ///< pointer to sensitivity of a wrt ux in sim_out
    double fun;                         ///< value of the cost function
} ocp_nlp_cost_external_memory;

//
acados_size_t ocp_nlp_cost_external_memory_calculate_size(void *config, void *dims, void *opts);
//
void *ocp_nlp_cost_external_memory_assign(void *config, void *dims, void *opts, void *raw_memory);
//
double *ocp_nlp_cost_external_memory_get_fun_ptr(void *memory_);
//
struct blasfeo_dvec *ocp_nlp_cost_external_memory_get_grad_ptr(void *memory_);
//
void ocp_nlp_cost_external_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory);
//
void ocp_nlp_cost_ls_memory_set_Z_ptr(struct blasfeo_dvec *Z, void *memory);
//
void ocp_nlp_cost_external_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_);
//
void ocp_nlp_cost_external_memory_set_z_alg_ptr(struct blasfeo_dvec *z_alg, void *memory_);
//
void ocp_nlp_cost_external_memory_set_dzdux_tran_ptr(struct blasfeo_dmat *dzdux_tran, void *memory_);
//
void ocp_nlp_cost_external_memory_set_jac_lag_stat_p_global_ptr(struct blasfeo_dmat *jac_lag_stat_p_global, void *memory_);

/************************************************
 * workspace
 ************************************************/

typedef struct
{
    struct blasfeo_dmat cost_grad_params_jac;  // jacobian of gradient of cost function wrt parameters
    struct blasfeo_dmat tmp_nunx_nunx;
    struct blasfeo_dmat tmp_nz_nz;
    struct blasfeo_dmat tmp_nz_nunx;
    struct blasfeo_dvec tmp_nunxnz;
    struct blasfeo_dvec tmp_2ns;  // temporary vector of dimension 2*ns
} ocp_nlp_cost_external_workspace;

//
acados_size_t ocp_nlp_cost_external_workspace_calculate_size(void *config, void *dims, void *opts);
//
size_t ocp_nlp_cost_external_get_external_fun_workspace_requirement(void *config_, void *dims_, void *opts_, void *model_);
//
void ocp_nlp_cost_external_set_external_fun_workspaces(void *config_, void *dims_, void *opts_, void *model_, void *workspace_);


/************************************************
 * functions
 ************************************************/

//
void ocp_nlp_cost_external_precompute(void *config_, void *dims_, void *model_, void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_external_config_initialize_default(void *config, int stage);
//
void ocp_nlp_cost_external_initialize(void *config_, void *dims, void *model_,
                                      void *opts_, void *mem_, void *work_);
//
void ocp_nlp_cost_external_update_qp_matrices(void *config_, void *dims, void *model_,
                                               void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_external_compute_fun(void *config_, void *dims, void *model_,
                                       void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_external_compute_jac_p(void *config_, void *dims, void *model_,
                                       void *opts_, void *memory_, void *work_);

void ocp_nlp_cost_external_compute_gradient(void *config_, void *dims, void *model_,
                                       void *opts_, void *memory_, void *work_);

void ocp_nlp_cost_external_eval_grad_p(void *config_, void *dims, void *model_, void *opts_, void *memory_, void *work_, struct blasfeo_dvec *out);
#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_COST_EXTERNAL_H_
