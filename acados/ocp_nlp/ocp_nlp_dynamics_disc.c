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


#include "acados/ocp_nlp/ocp_nlp_dynamics_disc.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// blasfeo
#include "blasfeo_d_aux.h"
#include "blasfeo_d_blas.h"
// acados
#include "acados/utils/mem.h"



/************************************************
 * dims
 ************************************************/

acados_size_t ocp_nlp_dynamics_disc_dims_calculate_size(void *config_)
{
    acados_size_t size = 0;

    size += sizeof(ocp_nlp_dynamics_disc_dims);

    return size;
}



void *ocp_nlp_dynamics_disc_dims_assign(void *config_, void *raw_memory)
{
    char *c_ptr = (char *) raw_memory;

    ocp_nlp_dynamics_disc_dims *dims = (ocp_nlp_dynamics_disc_dims *) c_ptr;
    c_ptr += sizeof(ocp_nlp_dynamics_disc_dims);

    dims->np = 0;
    dims->np_global = 0;

    assert((char *) raw_memory + ocp_nlp_dynamics_disc_dims_calculate_size(config_) >= c_ptr);

    return dims;
}



void ocp_nlp_dynamics_disc_dims_set(void *config_, void *dims_, const char *dim, int* value)
{
    ocp_nlp_dynamics_disc_dims *dims = (ocp_nlp_dynamics_disc_dims *) dims_;

    if (!strcmp(dim, "nx"))
    {
        dims->nx = *value;
    }
    else if (!strcmp(dim, "nx1"))
    {
        dims->nx1 = *value;
    }
    else if (!strcmp(dim, "nz"))
    {
        if ( *value > 0)
        {
            printf("\nerror: discrete dynamics with nz>0\n");
            exit(1);
        }
    }
    else if (!strcmp(dim, "nu"))
    {
        dims->nu = *value;
    }
    else if (!strcmp(dim, "nu1"))
    {
        dims->nu1 = *value;
    }
    else if (!strcmp(dim, "np"))
    {
        dims->np = *value;
    }
    else if (!strcmp(dim, "np_global"))
    {
        dims->np_global = *value;
    }
    else
    {
        printf("\ndimension type %s not available in module ocp_nlp_dynamics_disc\n", dim);
        exit(1);
    }
}

void ocp_nlp_dynamics_disc_dims_get(void *config_, void *dims_, const char *dim, int* value)
{
    ocp_nlp_dynamics_disc_dims *dims = (ocp_nlp_dynamics_disc_dims *) dims_;

    if (!strcmp(dim, "nx"))
    {
        *value = dims->nx;
    }
    else if (!strcmp(dim, "nx1"))
    {
        *value = dims->nx1;
    }
    else if (!strcmp(dim, "nz"))
    {
        if ( *value > 0)
        {
            printf("\nerror: ocp_nlp_dynamics_disc does not support nz > 0\n");
            exit(1);
        }
    }
    else if (!strcmp(dim, "nu"))
    {
        *value = dims->nu;
    }
    else if (!strcmp(dim, "nu1"))
    {
        *value = dims->nu1;
    }
    else if (!strcmp(dim, "np"))
    {
        *value = dims->np;
    }
    else if (!strcmp(dim, "np_global"))
    {
        *value = dims->np_global;
    }
    else
    {
        printf("\ndimension type %s not available in module ocp_nlp_dynamics_disc\n", dim);
        exit(1);
    }
}


/************************************************
 * options
 ************************************************/

acados_size_t ocp_nlp_dynamics_disc_opts_calculate_size(void *config_, void *dims_)
{
    // ocp_nlp_dynamics_config *config = config_;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_dynamics_disc_opts);

    return size;
}



void *ocp_nlp_dynamics_disc_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    // ocp_nlp_dynamics_config *config = config_;
    // ocp_nlp_dynamics_disc_dims *dims = dims_;

    char *c_ptr = (char *) raw_memory;

    ocp_nlp_dynamics_disc_opts *opts = (ocp_nlp_dynamics_disc_opts *) c_ptr;
    c_ptr += sizeof(ocp_nlp_dynamics_disc_opts);

    assert((char *) raw_memory + ocp_nlp_dynamics_disc_opts_calculate_size(config_, dims_) >=
           c_ptr);

    return opts;
}



void ocp_nlp_dynamics_disc_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    ocp_nlp_dynamics_disc_opts *opts = opts_;

    opts->compute_adj = 1;
    opts->compute_hess = 0;
    opts->cost_computation = 0;

    return;
}



void ocp_nlp_dynamics_disc_opts_update(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_dynamics_config *config = config_;
    // ocp_nlp_dynamics_disc_opts *opts = opts_;

    return;
}



void ocp_nlp_dynamics_disc_opts_set(void *config_, void *opts_, const char *field, void* value)
{

    ocp_nlp_dynamics_disc_opts *opts = opts_;

    if(!strcmp(field, "compute_adj"))
    {
        int *int_ptr = value;
        opts->compute_adj = *int_ptr;
    }
    else if(!strcmp(field, "compute_hess"))
    {
        int *int_ptr = value;
        opts->compute_hess = *int_ptr;
    }
    else if(!strcmp(field, "with_solution_sens_wrt_params"))
    {
        int *int_ptr = value;
        opts->with_solution_sens_wrt_params = *int_ptr;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_dynamics_disc_opts_set\n", field);
        exit(1);
    }

    return;

}


void ocp_nlp_dynamics_disc_opts_get(void *config_, void *opts_, const char *field, void* value)
{

    ocp_nlp_dynamics_disc_opts *opts = opts_;

    if (!strcmp(field, "compute_adj"))
    {
        int *int_ptr = value;
        *int_ptr = opts->compute_adj;
    }
    else if (!strcmp(field, "cost_computation"))
    {
        int *int_ptr = value;
        *int_ptr = opts->cost_computation;
    }
    else if (!strcmp(field, "compute_hess"))
    {
        int *int_ptr = value;
        *int_ptr = opts->compute_hess;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_dynamics_disc_opts_get\n", field);
        exit(1);
    }

    return;

}


/************************************************
 * memory
 ************************************************/

acados_size_t ocp_nlp_dynamics_disc_memory_calculate_size(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;

    // extract dims
    int nx = dims->nx;
    int nu = dims->nu;
    int nx1 = dims->nx1;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_dynamics_disc_memory);

    size += 1 * blasfeo_memsize_dvec(nu + nx + nx1);  // adj
    size += 1 * blasfeo_memsize_dvec(nx1);            // fun

    size += 64;  // blasfeo_mem align

    return size;
}



void *ocp_nlp_dynamics_disc_memory_assign(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;

    char *c_ptr = (char *) raw_memory;

    // extract dims
    int nx = dims->nx;
    int nu = dims->nu;
    int nx1 = dims->nx1;

    // struct
    ocp_nlp_dynamics_disc_memory *memory = (ocp_nlp_dynamics_disc_memory *) c_ptr;
    c_ptr += sizeof(ocp_nlp_dynamics_disc_memory);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);

    // adj
    assign_and_advance_blasfeo_dvec_mem(nu + nx + nx1, &memory->adj, &c_ptr);
    // fun
    assign_and_advance_blasfeo_dvec_mem(nx1, &memory->fun, &c_ptr);

    assert((char *) raw_memory +
               ocp_nlp_dynamics_disc_memory_calculate_size(config_, dims, opts_) >=
           c_ptr);

    return memory;
}



struct blasfeo_dvec *ocp_nlp_dynamics_disc_memory_get_fun_ptr(void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    return &memory->fun;
}



struct blasfeo_dvec *ocp_nlp_dynamics_disc_memory_get_adj_ptr(void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    return &memory->adj;
}



void ocp_nlp_dynamics_disc_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    memory->ux = ux;

    return;
}


void ocp_nlp_dynamics_disc_memory_set_ux1_ptr(struct blasfeo_dvec *ux1, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    memory->ux1 = ux1;

    return;
}


void ocp_nlp_dynamics_disc_memory_set_pi_ptr(struct blasfeo_dvec *pi, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    memory->pi = pi;

    return;
}




void ocp_nlp_dynamics_disc_memory_set_BAbt_ptr(struct blasfeo_dmat *BAbt, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    memory->BAbt = BAbt;

    return;
}



void ocp_nlp_dynamics_disc_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;

    memory->RSQrq = RSQrq;

    return;
}


void ocp_nlp_dynamics_disc_memory_set_dyn_jac_p_global_ptr(struct blasfeo_dmat *dyn_jac_p_global, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;
    memory->dyn_jac_p_global = dyn_jac_p_global;
}

void ocp_nlp_dynamics_disc_memory_set_jac_lag_stat_p_global_ptr(struct blasfeo_dmat *jac_lag_stat_p_global, void *memory_)
{
    ocp_nlp_dynamics_disc_memory *memory = memory_;
    memory->jac_lag_stat_p_global = jac_lag_stat_p_global;
}


void ocp_nlp_dynamics_disc_memory_set_dzduxt_ptr(struct blasfeo_dmat *mat, void *memory_)
{
    return;  // we don't allow algebraic variables for discrete models for now
}



void ocp_nlp_dynamics_disc_memory_set_sim_guess_ptr(struct blasfeo_dvec *z, bool *bool_ptr, void *memory_)
{
    return;  // we don't allow algebraic variables for discrete models for now
}



void ocp_nlp_dynamics_disc_memory_set_z_alg_ptr(struct blasfeo_dvec *z, void *memory_)
{
    return;  // we don't allow algebraic variables for discrete models for now
}



void ocp_nlp_dynamics_disc_memory_get(void *config_, void *dims_, void *mem_, const char *field, void* value)
{
//    ocp_nlp_dynamics_disc_dims *dims = dims_;
//    ocp_nlp_dynamics_disc_memory *mem = mem_;

    if (!strcmp(field, "time_sim") || !strcmp(field, "time_sim_ad") || !strcmp(field, "time_sim_la"))
    {
        double *ptr = value;
        *ptr = 0;
    }
    else
    {
        printf("\nerror: ocp_nlp_dynamics_disc_memory_get: field %s not available\n", field);
        exit(1);
    }

}


/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_nlp_dynamics_disc_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    // ocp_nlp_dynamics_disc_opts *opts = opts_;

    int nx = dims->nx;
    int nu = dims->nu;
    // int nx1 = dims->nx1;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_dynamics_disc_workspace);

    size += 1 * blasfeo_memsize_dmat(nu+nx, nu+nx);   // tmp_nv_nv
    size += 1*64;  // blasfeo_mem align

    return size;
}



static void ocp_nlp_dynamics_disc_cast_workspace(void *config_, void *dims_, void *opts_,
                                                 void *work_)
{
    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    // ocp_nlp_dynamics_disc_opts *opts = opts_;
    ocp_nlp_dynamics_disc_workspace *work = work_;

    int nx = dims->nx;
    int nu = dims->nu;
    // int nx1 = dims->nx1;

    char *c_ptr = (char *) work_;
    c_ptr += sizeof(ocp_nlp_dynamics_disc_workspace);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);

    // tmp_nv_nv
    assign_and_advance_blasfeo_dmat_mem(nu+nx, nu+nx, &work->tmp_nv_nv, &c_ptr);

    assert((char *) work + ocp_nlp_dynamics_disc_workspace_calculate_size(config_, dims, opts_) >= c_ptr);

    return;
}



/************************************************
 * model
 ************************************************/

acados_size_t ocp_nlp_dynamics_disc_model_calculate_size(void *config_, void *dims_)
{
    // ocp_nlp_dynamics_config *config = config_;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_dynamics_disc_model);

    return size;
}



void *ocp_nlp_dynamics_disc_model_assign(void *config_, void *dims_, void *raw_memory)
{

    char *c_ptr = (char *) raw_memory;

    // struct
    ocp_nlp_dynamics_disc_model *model = (ocp_nlp_dynamics_disc_model *) c_ptr;
    c_ptr += sizeof(ocp_nlp_dynamics_disc_model);

    assert((char *) raw_memory + ocp_nlp_dynamics_disc_model_calculate_size(config_, dims_) >=
           c_ptr);

    return model;
}



void ocp_nlp_dynamics_disc_model_set(void *config_, void *dims_, void *model_, const char *field, void *value)
{

    ocp_nlp_dynamics_disc_model *model = model_;

    if (!strcmp(field, "T"))
    {
        // do nothing
    }
    else if (!strcmp(field, "disc_dyn_fun"))
    {
        model->disc_dyn_fun = (external_function_generic *) value;
    }
    else if (!strcmp(field, "disc_dyn_fun_jac"))
    {
        model->disc_dyn_fun_jac = (external_function_generic *) value;
    }
    else if (!strcmp(field, "disc_dyn_fun_jac_hess"))
    {
        model->disc_dyn_fun_jac_hess = (external_function_generic *) value;
    }
    else if (!strcmp(field, "disc_dyn_phi_jac_p_hess_xu_p"))
    {
        model->disc_dyn_phi_jac_p_hess_xu_p = (external_function_generic *) value;
    }
    else if (!strcmp(field, "disc_dyn_adj_p"))
    {
        model->disc_dyn_adj_p = (external_function_generic *) value;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_dynamics_disc_model_set\n", field);
        exit(1);
    }

    return;
}



/************************************************
 * functions
 ************************************************/

void ocp_nlp_dynamics_disc_initialize(void *config_, void *dims_, void *model_, void *opts_,
                                      void *mem_, void *work_)
{
    return;
}



void ocp_nlp_dynamics_disc_update_qp_matrices(void *config_, void *dims_, void *model_, void *opts_,
                                              void *mem_, void *work_)
{
    ocp_nlp_dynamics_disc_cast_workspace(config_, dims_, opts_, work_);

    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    ocp_nlp_dynamics_disc_opts *opts = opts_;
    // ocp_nlp_dynamics_disc_workspace *work = work_;
    ocp_nlp_dynamics_disc_memory *memory = mem_;
    ocp_nlp_dynamics_disc_model *model = model_;

    int nx = dims->nx;
    int nu = dims->nu;
    int nx1 = dims->nx1;
    int nu1 = dims->nu1;

    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[3];
    void *ext_fun_out[3];

    // pass state and control to integrator
    struct blasfeo_dvec_args x_in;  // input x of external fun;
    x_in.x = memory->ux;
    x_in.xi = nu;

    struct blasfeo_dvec_args u_in;  // input u of external fun;
    u_in.x = memory->ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args fun_out;
    fun_out.x = &memory->fun;
    fun_out.xi = 0;

    struct blasfeo_dmat_args jac_out;
    jac_out.A = memory->BAbt;
    jac_out.ai = 0;
    jac_out.aj = 0;

    if (opts->compute_hess)
    {
        struct blasfeo_dvec_args pi_in;  // input u of external fun;
        pi_in.x = memory->pi;
        pi_in.xi = 0;

        struct blasfeo_dmat_args hess_out;
        hess_out.A = memory->RSQrq;
        hess_out.ai = 0;
        hess_out.aj = 0;

        ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
        ext_fun_in[0] = &x_in;
        ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
        ext_fun_in[1] = &u_in;
        ext_fun_type_in[2] = BLASFEO_DVEC_ARGS;
        ext_fun_in[2] = &pi_in;

        ext_fun_type_out[0] = BLASFEO_DVEC_ARGS;
        ext_fun_out[0] = &fun_out;  // fun: nx1
        ext_fun_type_out[1] = BLASFEO_DMAT_ARGS;
        ext_fun_out[1] = &jac_out;  // jac': (nu+nx) * nx1
        ext_fun_type_out[2] = BLASFEO_DMAT_ARGS;
        ext_fun_out[2] = &hess_out;  // hess*pi: (nu+nx)*(nu+nx)

        // call external function
        model->disc_dyn_fun_jac_hess->evaluate(model->disc_dyn_fun_jac_hess, ext_fun_type_in, ext_fun_in,
                ext_fun_type_out, ext_fun_out);
    }
    else
    {
        ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
        ext_fun_in[0] = &x_in;
        ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
        ext_fun_in[1] = &u_in;

        ext_fun_type_out[0] = BLASFEO_DVEC_ARGS;
        ext_fun_out[0] = &fun_out;  // fun: nx1
        ext_fun_type_out[1] = BLASFEO_DMAT_ARGS;
        ext_fun_out[1] = &jac_out;  // jac': (nu+nx) * nx1

        // call external function
        model->disc_dyn_fun_jac->evaluate(model->disc_dyn_fun_jac, ext_fun_type_in, ext_fun_in, ext_fun_type_out, ext_fun_out);
    }

    // fun
    blasfeo_daxpy(nx1, -1.0, memory->ux1, nu1, &memory->fun, 0, &memory->fun, 0);

    // adj TODO if not computed by the external function
    if (opts->compute_adj)
    {
        blasfeo_dgemv_n(nu+nx, nx1, -1.0, memory->BAbt, 0, 0, memory->pi, 0, 0.0, &memory->adj, 0, &memory->adj, 0);
        blasfeo_dveccp(nx1, memory->pi, 0, &memory->adj, nu + nx);
    }

    return;
}



void ocp_nlp_dynamics_disc_compute_fun(void *config_, void *dims_, void *model_, void *opts_,
                                              void *mem_, void *work_)
{
    ocp_nlp_dynamics_disc_cast_workspace(config_, dims_, opts_, work_);

    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    // ocp_nlp_dynamics_disc_opts *opts = opts_;
    // ocp_nlp_dynamics_disc_workspace *work = work_;
    ocp_nlp_dynamics_disc_memory *memory = mem_;
    ocp_nlp_dynamics_disc_model *model = model_;

    // int nx = dims->nx;
    int nu = dims->nu;
    int nx1 = dims->nx1;
    int nu1 = dims->nu1;

    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[1];
    void *ext_fun_out[1];

    struct blasfeo_dvec *ux = memory->ux;
    struct blasfeo_dvec *ux1 = memory->ux1;

    // pass state and control to integrator
    struct blasfeo_dvec_args x_in;  // input x of external fun;
    x_in.x = ux;
    x_in.xi = nu;

    struct blasfeo_dvec_args u_in;  // input u of external fun;
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args fun_out;
    fun_out.x = &memory->fun;
    fun_out.xi = 0;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;

    ext_fun_type_out[0] = BLASFEO_DVEC_ARGS;
    ext_fun_out[0] = &fun_out;  // fun: nx1

    // call external function
    model->disc_dyn_fun->evaluate(model->disc_dyn_fun, ext_fun_type_in, ext_fun_in, ext_fun_type_out, ext_fun_out);

    // fun
    blasfeo_daxpy(nx1, -1.0, ux1, nu1, &memory->fun, 0, &memory->fun, 0);

    return;
}

void ocp_nlp_dynamics_disc_compute_jac_hess_p(void *config_, void *dims_, void *model_, void *opts_, void *mem_, void *work_)
{
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    ocp_nlp_dynamics_disc_memory *memory = mem_;
    ocp_nlp_dynamics_disc_model *model = model_;

    int nu = dims->nu;

    ext_fun_arg_t ext_fun_type_in[4];
    void *ext_fun_in[4];
    ext_fun_arg_t ext_fun_type_out[2];
    void *ext_fun_out[2];

    struct blasfeo_dvec *ux = memory->ux;

    struct blasfeo_dvec_args x_in;  // input x of external fun;
    x_in.x = ux;
    x_in.xi = nu;

    struct blasfeo_dvec_args u_in;  // input u of external fun;
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args pi_in; // input pi of external fun;
    pi_in.x = memory->pi;
    pi_in.xi = 0;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;

    ext_fun_type_in[2] = BLASFEO_DVEC_ARGS;
    ext_fun_in[2] = &pi_in;

    ext_fun_type_out[0] = BLASFEO_DMAT;
    ext_fun_out[0] = memory->dyn_jac_p_global;  // jac: nx1 x np_global

    ext_fun_type_out[1] = BLASFEO_DMAT_ARGS;
    struct blasfeo_dmat_args lag_stat_jac_p_global_out; // input pi of external fun;
    ext_fun_out[1] = &lag_stat_jac_p_global_out;
    lag_stat_jac_p_global_out.A = memory->jac_lag_stat_p_global;  // jac: nxnu x np_global
    lag_stat_jac_p_global_out.ai = 0;
    lag_stat_jac_p_global_out.aj = 0;

    // call external function
    model->disc_dyn_phi_jac_p_hess_xu_p->evaluate(model->disc_dyn_phi_jac_p_hess_xu_p, ext_fun_type_in, ext_fun_in, ext_fun_type_out, ext_fun_out);

    return;
}


void ocp_nlp_dynamics_disc_compute_fun_and_adj(void *config_, void *dims_, void *model_, void *opts_,
                                              void *mem_, void *work_)
{
    /* TODO: this is inefficient! Generate a separate function for discrete dynamics to compute fun and adj! */
    // when this is done tmp_nv_nv can be removed from work if compute_hess is false.
    ocp_nlp_dynamics_disc_cast_workspace(config_, dims_, opts_, work_);

    // ocp_nlp_dynamics_config *config = config_;
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    ocp_nlp_dynamics_disc_opts *opts = opts_;
    ocp_nlp_dynamics_disc_workspace *work = work_;
    ocp_nlp_dynamics_disc_memory *memory = mem_;
    ocp_nlp_dynamics_disc_model *model = model_;

    int nx = dims->nx;
    int nu = dims->nu;
    int nx1 = dims->nx1;
    int nu1 = dims->nu1;

    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[3];
    void *ext_fun_out[3];

    // pass state and control to integrator
    struct blasfeo_dvec_args x_in;  // input x of external fun;
    x_in.x = memory->ux;
    x_in.xi = nu;

    struct blasfeo_dvec_args u_in;  // input u of external fun;
    u_in.x = memory->ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args fun_out;
    fun_out.x = &memory->fun;
    fun_out.xi = 0;

    struct blasfeo_dmat_args jac_out;
    jac_out.A = &work->tmp_nv_nv;
    jac_out.ai = 0;
    jac_out.aj = 0;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;

    ext_fun_type_out[0] = BLASFEO_DVEC_ARGS;
    ext_fun_out[0] = &fun_out;  // fun: nx1
    ext_fun_type_out[1] = BLASFEO_DMAT_ARGS;
    ext_fun_out[1] = &jac_out;  // jac': (nu+nx) * nx1

    // call external function
    model->disc_dyn_fun_jac->evaluate(model->disc_dyn_fun_jac, ext_fun_type_in, ext_fun_in, ext_fun_type_out, ext_fun_out);

    // fun
    blasfeo_daxpy(nx1, -1.0, memory->ux1, nu1, &memory->fun, 0, &memory->fun, 0);

    // adj TODO if not computed by the external function
    if (opts->compute_adj)
    {
        blasfeo_dgemv_n(nu+nx, nx1, -1.0, &work->tmp_nv_nv, 0, 0, memory->pi, 0, 0.0, &memory->adj, 0, &memory->adj, 0);
        blasfeo_dveccp(nx1, memory->pi, 0, &memory->adj, nu + nx);
    }

    return;
}


int ocp_nlp_dynamics_disc_precompute(void *config_, void *dims, void *model_, void *opts_,
                                        void *mem_, void *work_)
{
    return ACADOS_SUCCESS;
}

void ocp_nlp_dynamics_disc_compute_adj_p(void* config_, void *dims_, void *model_, void *opts_, void *mem_, struct blasfeo_dvec *out)
{
    ocp_nlp_dynamics_disc_dims *dims = dims_;
    ocp_nlp_dynamics_disc_memory *memory = mem_;
    ocp_nlp_dynamics_disc_model *model = model_;

    int nu = dims->nu;

    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[1];
    void *ext_fun_out[1];

    struct blasfeo_dvec *ux = memory->ux;

    struct blasfeo_dvec_args x_in;  // input x of external fun;
    x_in.x = ux;
    x_in.xi = nu;

    struct blasfeo_dvec_args u_in;  // input u of external fun;
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args pi_in; // input pi of external fun;
    pi_in.x = memory->pi;
    pi_in.xi = 0;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;

    ext_fun_type_in[2] = BLASFEO_DVEC_ARGS;
    ext_fun_in[2] = &pi_in;

    ext_fun_type_out[0] = BLASFEO_DVEC;
    ext_fun_out[0] = out;

    // call external function
    if (model->disc_dyn_adj_p == NULL)
    {
        printf("ocp_nlp_dynamics_disc_compute_adj_p - model->disc_dyn_adj_p is NULL\n");
        exit(1);
    }
    model->disc_dyn_adj_p->evaluate(model->disc_dyn_adj_p, ext_fun_type_in, ext_fun_in, ext_fun_type_out, ext_fun_out);

    return;
}



size_t ocp_nlp_dynamics_disc_get_external_fun_workspace_requirement(void *config_, void *dims_, void *opts_, void *model_)
{
    ocp_nlp_dynamics_disc_model *model = model_;
    // ocp_nlp_dynamics_config *config = config_;
    // ocp_nlp_dynamics_disc_opts *opts = opts_;
    // ocp_nlp_dynamics_disc_dims *dims = dims_;

    size_t size = 0;
    size_t tmp_size;

    tmp_size = external_function_get_workspace_requirement_if_defined(model->disc_dyn_fun);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->disc_dyn_adj_p);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->disc_dyn_fun_jac);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->disc_dyn_fun_jac_hess);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->disc_dyn_phi_jac_p_hess_xu_p);
    size = size > tmp_size ? size : tmp_size;

    return size;
}


void ocp_nlp_dynamics_disc_set_external_fun_workspaces(void *config_, void *dims_, void *opts_, void *model_, void *workspace_)
{
    ocp_nlp_dynamics_disc_model *model = model_;

    external_function_set_fun_workspace_if_defined(model->disc_dyn_fun, workspace_);
    external_function_set_fun_workspace_if_defined(model->disc_dyn_adj_p, workspace_);
    external_function_set_fun_workspace_if_defined(model->disc_dyn_fun_jac, workspace_);
    external_function_set_fun_workspace_if_defined(model->disc_dyn_fun_jac_hess, workspace_);
    external_function_set_fun_workspace_if_defined(model->disc_dyn_phi_jac_p_hess_xu_p, workspace_);
}


void ocp_nlp_dynamics_disc_config_initialize_default(void *config_, int stage)
{
    ocp_nlp_dynamics_config *config = config_;

    config->dims_calculate_size = &ocp_nlp_dynamics_disc_dims_calculate_size;
    config->dims_assign = &ocp_nlp_dynamics_disc_dims_assign;
    config->dims_set =  &ocp_nlp_dynamics_disc_dims_set;
    config->dims_get = &ocp_nlp_dynamics_disc_dims_get;
    config->model_calculate_size = &ocp_nlp_dynamics_disc_model_calculate_size;
    config->model_assign = &ocp_nlp_dynamics_disc_model_assign;
    config->model_set = &ocp_nlp_dynamics_disc_model_set;
    config->opts_calculate_size = &ocp_nlp_dynamics_disc_opts_calculate_size;
    config->opts_assign = &ocp_nlp_dynamics_disc_opts_assign;
    config->opts_initialize_default = &ocp_nlp_dynamics_disc_opts_initialize_default;
    config->opts_update = &ocp_nlp_dynamics_disc_opts_update;
    config->opts_set = &ocp_nlp_dynamics_disc_opts_set;
    config->opts_get = &ocp_nlp_dynamics_disc_opts_get;
    config->memory_calculate_size = &ocp_nlp_dynamics_disc_memory_calculate_size;
    config->memory_assign = &ocp_nlp_dynamics_disc_memory_assign;
    config->memory_get_fun_ptr = &ocp_nlp_dynamics_disc_memory_get_fun_ptr;
    config->memory_get_adj_ptr = &ocp_nlp_dynamics_disc_memory_get_adj_ptr;
    config->memory_set_ux_ptr = &ocp_nlp_dynamics_disc_memory_set_ux_ptr;
    config->memory_set_ux1_ptr = &ocp_nlp_dynamics_disc_memory_set_ux1_ptr;
    config->memory_set_pi_ptr = &ocp_nlp_dynamics_disc_memory_set_pi_ptr;
    config->memory_set_BAbt_ptr = &ocp_nlp_dynamics_disc_memory_set_BAbt_ptr;
    config->memory_set_RSQrq_ptr = &ocp_nlp_dynamics_disc_memory_set_RSQrq_ptr;
    config->memory_set_dzduxt_ptr = &ocp_nlp_dynamics_disc_memory_set_dzduxt_ptr;
    config->memory_set_sim_guess_ptr = &ocp_nlp_dynamics_disc_memory_set_sim_guess_ptr;
    config->memory_set_z_alg_ptr = &ocp_nlp_dynamics_disc_memory_set_z_alg_ptr;
    config->memory_set_dyn_jac_p_global_ptr = &ocp_nlp_dynamics_disc_memory_set_dyn_jac_p_global_ptr;
    config->memory_get = &ocp_nlp_dynamics_disc_memory_get;
    config->memory_set_jac_lag_stat_p_global_ptr = &ocp_nlp_dynamics_disc_memory_set_jac_lag_stat_p_global_ptr;
    config->memory_set_dyn_jac_p_global_ptr = &ocp_nlp_dynamics_disc_memory_set_dyn_jac_p_global_ptr;
    config->compute_jac_hess_p = &ocp_nlp_dynamics_disc_compute_jac_hess_p;
    config->workspace_calculate_size = &ocp_nlp_dynamics_disc_workspace_calculate_size;
    config->get_external_fun_workspace_requirement = &ocp_nlp_dynamics_disc_get_external_fun_workspace_requirement;
    config->set_external_fun_workspaces = &ocp_nlp_dynamics_disc_set_external_fun_workspaces;
    config->initialize = &ocp_nlp_dynamics_disc_initialize;
    config->update_qp_matrices = &ocp_nlp_dynamics_disc_update_qp_matrices;
    config->compute_fun = &ocp_nlp_dynamics_disc_compute_fun;
    config->compute_fun_and_adj = &ocp_nlp_dynamics_disc_compute_fun_and_adj;
    config->compute_adj_p = &ocp_nlp_dynamics_disc_compute_adj_p;
    config->precompute = &ocp_nlp_dynamics_disc_precompute;
    config->config_initialize_default = &ocp_nlp_dynamics_disc_config_initialize_default;
    config->stage = stage;

    return;
}
