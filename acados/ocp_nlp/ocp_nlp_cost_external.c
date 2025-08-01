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


#include "acados/ocp_nlp/ocp_nlp_cost_external.h"
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// blasfeo
#include "blasfeo_d_aux.h"
#include "blasfeo_d_blas.h"
// acados
#include "acados/utils/mem.h"
#include "acados/utils/print.h"



/************************************************
 * dims
 ************************************************/

acados_size_t ocp_nlp_cost_external_dims_calculate_size(void *config_)
{
    acados_size_t size = sizeof(ocp_nlp_cost_external_dims);

    return size;
}



void *ocp_nlp_cost_external_dims_assign(void *config_, void *raw_memory)
{
    char *c_ptr = (char *) raw_memory;

    ocp_nlp_cost_external_dims *dims = (ocp_nlp_cost_external_dims *) c_ptr;
    c_ptr += sizeof(ocp_nlp_cost_external_dims);

    dims->np = 0;
    dims->nz = 0;
    dims->ns = 0;
    dims->nu = 0;

    assert((char *) raw_memory + ocp_nlp_cost_external_dims_calculate_size(config_) >= c_ptr);

    return dims;
}



void ocp_nlp_cost_external_dims_set(void *config_, void *dims_, const char *field, int* value)
{
    ocp_nlp_cost_external_dims *dims = (ocp_nlp_cost_external_dims *) dims_;

    if (!strcmp(field, "nx"))
    {
        dims->nx = *value;
    }
    else if (!strcmp(field, "nz"))
    {
        dims->nz = *value;
    }
    else if (!strcmp(field, "nu"))
    {
        dims->nu = *value;
    }
    else if (!strcmp(field, "ns"))
    {
        dims->ns = *value;
    }
    else if (!strcmp(field, "np"))
    {
        dims->np = *value;
    }
    else if (!strcmp(field, "np_global"))
    {
        dims->np_global = *value;
    }
    else
    {
        printf("\nerror: ocp_nlp_cost_external_dims_set: dimension type %s not available.\n", field);
        exit(1);
    }

    return;
}



void ocp_nlp_cost_external_dims_get(void *config_, void *dims_, const char *field, int* value)
{
        printf("error: ocp_nlp_cost_external_dims_get: attempt to get dimensions of non-existing field %s\n", field);
        exit(1);
}



/************************************************
 * model
 ************************************************/

acados_size_t ocp_nlp_cost_external_model_calculate_size(void *config_, void *dims_)
{
    ocp_nlp_cost_external_dims *dims = dims_;

    int nx = dims->nx;
    int nu = dims->nu;
    int ns = dims->ns;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_cost_external_model);

    size += 1 * 64;  // blasfeo_mem align
    size += blasfeo_memsize_dmat(nx+nu, nx+nu);

    size += 2 * blasfeo_memsize_dvec(2 * ns);  // Z, z

    return size;
}



void *ocp_nlp_cost_external_model_assign(void *config_, void *dims_, void *raw_memory)
{
    ocp_nlp_cost_external_dims *dims = dims_;

    char *c_ptr = (char *) raw_memory;

    int nx = dims->nx;
    int nu = dims->nu;
    int ns = dims->ns;

    // struct
    ocp_nlp_cost_external_model *model = (ocp_nlp_cost_external_model *) c_ptr;
    c_ptr += sizeof(ocp_nlp_cost_external_model);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);
    // numerical_hessian
    assign_and_advance_blasfeo_dmat_mem(nx+nu, nx+nu, &model->numerical_hessian, &c_ptr);

    // blasfeo_dvec
    // Z
    assign_and_advance_blasfeo_dvec_mem(2 * ns, &model->Z, &c_ptr);
    // z
    assign_and_advance_blasfeo_dvec_mem(2 * ns, &model->z, &c_ptr);

    // default initialization
    model->scaling = 1.0;

    // assert
    assert((char *) raw_memory + ocp_nlp_cost_external_model_calculate_size(config_, dims_) >=
           c_ptr);

    return model;
}



int ocp_nlp_cost_external_model_set(void *config_, void *dims_, void *model_,
                                         const char *field, void *value_)
{
    int status = ACADOS_SUCCESS;

    if ( !config_ || !dims_ || !model_ || !value_ )
    {
        printf("ocp_nlp_cost_external_model_set: got Null pointer \n");
        exit(1);
    }

    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;

    int ns = dims->ns;
    int nx = dims->nx;
    int nu = dims->nu;

    if (!strcmp(field, "ext_cost_fun"))
    {
        model->ext_cost_fun = (external_function_generic *) value_;
    }
    else if (!strcmp(field, "ext_cost_fun_jac_hes") || !strcmp(field, "ext_cost_fun_jac_hess"))
    {
        model->ext_cost_fun_jac_hess = (external_function_generic *) value_;
    }
    else if (!strcmp(field, "ext_cost_fun_jac"))
    {
        model->ext_cost_fun_jac = (external_function_generic *) value_;
    }
    else if (!strcmp(field, "ext_cost_hess_xu_p"))
    {
        model->ext_cost_hess_xu_p = (external_function_generic *) value_;
    }
    else if (!strcmp(field, "ext_cost_grad_p"))
    {
        model->ext_cost_grad_p = (external_function_generic *) value_;
    }
    else if (!strcmp(field, "ext_cost_num_hess"))
    {
        double *numerical_hessian = (double *) value_;
        blasfeo_pack_dmat(nx+nu, nx+nu, numerical_hessian, nx+nu, &model->numerical_hessian, 0, 0);
    }
    else if (!strcmp(field, "Z"))
    {
        double *Z = (double *) value_;
        blasfeo_pack_dvec(ns, Z, 1, &model->Z, 0);
        blasfeo_pack_dvec(ns, Z, 1, &model->Z, ns);
    }
    else if (!strcmp(field, "Zl"))
    {
        double *Zl = (double *) value_;
        blasfeo_pack_dvec(ns, Zl, 1, &model->Z, 0);
    }
    else if (!strcmp(field, "Zu"))
    {
        double *Zu = (double *) value_;
        blasfeo_pack_dvec(ns, Zu, 1, &model->Z, ns);
    }
    else if (!strcmp(field, "z"))
    {
        double *z = (double *) value_;
        blasfeo_pack_dvec(ns, z, 1, &model->z, 0);
        blasfeo_pack_dvec(ns, z, 1, &model->z, ns);
    }
    else if (!strcmp(field, "zl"))
    {
        double *zl = (double *) value_;
        blasfeo_pack_dvec(ns, zl, 1, &model->z, 0);
    }
    else if (!strcmp(field, "zu"))
    {
        double *zu = (double *) value_;
        blasfeo_pack_dvec(ns, zu, 1, &model->z, ns);
    }
    else if (!strcmp(field, "scaling"))
    {
        double *scaling_ptr = (double *) value_;
        model->scaling = *scaling_ptr;
    }
    else
    {
        printf("\nerror: %s not available in module ocp_nlp_cost_external_model_set\n", field);
        exit(1);
    }
    return status;
}



int ocp_nlp_cost_external_model_get(void *config_, void *dims_, void *model_,
                                         const char *field, void *value_)
{
    int status = ACADOS_SUCCESS;

    if ( !config_ || !dims_ || !model_ || !value_ )
    {
        printf("ocp_nlp_cost_external_model_set: got Null pointer \n");
        exit(1);
    }

    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;

    int ns = dims->ns;
    int nx = dims->nx;
    int nu = dims->nu;

    double * value = (double *) value_;

    if (!strcmp(field, "ext_cost_num_hess"))
    {
        printf("in cost_get numerical hessian\n");
        blasfeo_unpack_dmat(nx+nu, nx+nu, &model->numerical_hessian, 0, 0, value, nx+nu);
    }
    else if (!strcmp(field, "Zl"))
    {
        blasfeo_unpack_dvec(ns, &model->Z, 0, value, 1);
    }
    else if (!strcmp(field, "Zu"))
    {
        blasfeo_unpack_dvec(ns, &model->Z, ns, value, 1);
    }
    else if (!strcmp(field, "zl"))
    {
        blasfeo_unpack_dvec(ns, &model->z, 0, value, 1);
    }
    else if (!strcmp(field, "zu"))
    {
        blasfeo_unpack_dvec(ns, &model->z, ns, value, 1);
    }
    else if (!strcmp(field, "scaling"))
    {
        value[0] = model->scaling;
    }
    else
    {
        printf("\nerror: %s not available in module ocp_nlp_cost_external_model_get\n", field);
        exit(1);
    }
    return status;
}

double *ocp_nlp_cost_external_model_get_scaling_ptr(void *in_)
{
    ocp_nlp_cost_external_model *model = in_;
    return &model->scaling;
}

/************************************************
 * options
 ************************************************/

acados_size_t ocp_nlp_cost_external_opts_calculate_size(void *config_, void *dims_)
{
    // ocp_nlp_cost_config *config = config_;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_cost_external_opts);
    make_int_multiple_of(8, &size);

    return size;
}



void *ocp_nlp_cost_external_opts_assign(void *config_, void *dims_, void *raw_memory)
{
    // ocp_nlp_cost_config *config = config_;

    char *c_ptr = (char *) raw_memory;

    ocp_nlp_cost_external_opts *opts = (ocp_nlp_cost_external_opts *) c_ptr;
    c_ptr += sizeof(ocp_nlp_cost_external_opts);

    assert((char *) raw_memory + ocp_nlp_cost_external_opts_calculate_size(config_, dims_) >=
           c_ptr);

    return opts;
}



void ocp_nlp_cost_external_opts_initialize_default(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_cost_config *config = config_;
    ocp_nlp_cost_external_opts *opts = opts_;

    opts->use_numerical_hessian = 0;
    opts->with_solution_sens_wrt_params = 0;
    opts->add_hess_contribution = 0;

    return;
}



void ocp_nlp_cost_external_opts_update(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_cost_config *config = config_;
    // ocp_nlp_cost_external_opts *opts = opts_;

    // opts->gauss_newton_hess = 1;

    return;
}



void ocp_nlp_cost_external_opts_set(void *config_, void *opts_, const char *field, void* value)
{
    // ocp_nlp_cost_config *config = config_;
    ocp_nlp_cost_external_opts *opts = opts_;

    if(!strcmp(field, "exact_hess"))
    {
        // do nothing: the exact hessian is always computed if no custom hessian is provided
    }
    else if(!strcmp(field, "numerical_hessian"))
    {
        int *opt_val = (int *) value;
        opts->use_numerical_hessian = *opt_val;
    }
    else if (!strcmp(field, "add_hess_contribution"))
    {
        int* int_ptr = value;
        opts->add_hess_contribution = *int_ptr;
    }
    else if(!strcmp(field, "with_solution_sens_wrt_params"))
    {
        int *opt_val = (int *) value;
        opts->with_solution_sens_wrt_params = *opt_val;
    }
    else
    {
        printf("\nerror: field %s not available in ocp_nlp_cost_external_opts_set\n", field);
        exit(1);
    }

    return;

}

int* ocp_nlp_cost_external_opts_get_add_hess_contribution_ptr(void *config_, void *opts_)
{
    ocp_nlp_cost_external_opts *opts = opts_;

    return &opts->add_hess_contribution;
}

/************************************************
 * memory
 ************************************************/

acados_size_t ocp_nlp_cost_external_memory_calculate_size(void *config_, void *dims_, void *opts_)
{
    // ocp_nlp_cost_config *config = config_;
    ocp_nlp_cost_external_dims *dims = dims_;

    int nx = dims->nx;
    int nu = dims->nu;
    int ns = dims->ns;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_cost_external_memory);

    size += 1 * blasfeo_memsize_dvec(nu + nx + 2 * ns);  // grad

    size += 64;  // blasfeo_mem align

    return size;
}



void *ocp_nlp_cost_external_memory_assign(void *config_, void *dims_, void *opts_, void *raw_memory)
{
    // ocp_nlp_cost_config *config = config_;
    ocp_nlp_cost_external_dims *dims = dims_;

    char *c_ptr = (char *) raw_memory;

    // extract dims
    int nx = dims->nx;
    int nu = dims->nu;
    int ns = dims->ns;

    // struct
    ocp_nlp_cost_external_memory *memory = (ocp_nlp_cost_external_memory *) c_ptr;
    c_ptr += sizeof(ocp_nlp_cost_external_memory);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);

    // grad
    assign_and_advance_blasfeo_dvec_mem(nu + nx + 2 * ns, &memory->grad, &c_ptr);

    assert((char *) raw_memory +
               ocp_nlp_cost_external_memory_calculate_size(config_, dims, opts_) >=
           c_ptr);

    return memory;
}



double *ocp_nlp_cost_external_memory_get_fun_ptr(void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    return &memory->fun;
}



struct blasfeo_dvec *ocp_nlp_cost_external_memory_get_grad_ptr(void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    return &memory->grad;
}



void ocp_nlp_cost_external_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->RSQrq = RSQrq;

    return;
}



void ocp_nlp_cost_external_memory_set_Z_ptr(struct blasfeo_dvec *Z, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->Z = Z;
}



void ocp_nlp_cost_external_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->ux = ux;

    return;
}


void ocp_nlp_cost_external_memory_set_z_alg_ptr(struct blasfeo_dvec *z_alg, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->z_alg = z_alg;
}



void ocp_nlp_cost_external_memory_set_dzdux_tran_ptr(struct blasfeo_dmat *dzdux_tran, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->dzdux_tran = dzdux_tran;
}



void ocp_nlp_cost_external_memory_set_jac_lag_stat_p_global_ptr(struct blasfeo_dmat *jac_lag_stat_p_global, void *memory_)
{
    ocp_nlp_cost_external_memory *memory = memory_;

    memory->jac_lag_stat_p_global = jac_lag_stat_p_global;
}


/************************************************
 * workspace
 ************************************************/

acados_size_t ocp_nlp_cost_external_workspace_calculate_size(void *config_, void *dims_, void *opts_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_opts *opts = opts_;

    // extract dims
    int nx = dims->nx;
    int nz = dims->nz;
    int nu = dims->nu;
    int ns = dims->ns;
    int np_global = dims->np_global;

    acados_size_t size = 0;

    size += sizeof(ocp_nlp_cost_external_workspace);

    if (opts->with_solution_sens_wrt_params)
    {
        size += 1 * blasfeo_memsize_dmat(nu + nx, np_global);  // cost_grad_params_jac
    }
    size += 1 * blasfeo_memsize_dmat(nu+nx, nu+nx);  // tmp_nunx_nunx
    size += 1 * blasfeo_memsize_dmat(nz, nz);  // tmp_nz_nz
    size += 1 * blasfeo_memsize_dmat(nz, nu+nx);  // tmp_nz_nunx
    size += 1 * blasfeo_memsize_dvec(nu+nx+nz);  // tmp_nunxnz

    size += 1 * blasfeo_memsize_dvec(2*ns);  // tmp_2ns

    size += 64;  // blasfeo_mem align

    return size;
}



static void ocp_nlp_cost_external_cast_workspace(void *config_, void *dims_, void *opts_,
                                                 void *work_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_workspace *work = work_;
    ocp_nlp_cost_external_opts *opts = opts_;

    // extract dims
    int nx = dims->nx;
    int nz = dims->nz;
    int nu = dims->nu;
    int ns = dims->ns;
    int np_global = dims->np_global;

    char *c_ptr = (char *) work_;
    c_ptr += sizeof(ocp_nlp_cost_external_workspace);

    // blasfeo_mem align
    align_char_to(64, &c_ptr);


    if (opts->with_solution_sens_wrt_params)
    {
        assign_and_advance_blasfeo_dmat_mem(nu + nx, np_global, &work->cost_grad_params_jac, &c_ptr);
    }

    // tmp_nunx_nunx
    assign_and_advance_blasfeo_dmat_mem(nu + nx, nu + nx, &work->tmp_nunx_nunx, &c_ptr);

    // tmp_nz_nz
    assign_and_advance_blasfeo_dmat_mem(nz, nz, &work->tmp_nz_nz, &c_ptr);

    // tmp_nz_nunx
    assign_and_advance_blasfeo_dmat_mem(nz, nu+nx, &work->tmp_nz_nunx, &c_ptr);

    // tmp_nunxnz
    assign_and_advance_blasfeo_dvec_mem(nu + nx + nz, &work->tmp_nunxnz, &c_ptr);

    // tmp_2ns
    assign_and_advance_blasfeo_dvec_mem(2*ns, &work->tmp_2ns, &c_ptr);

    assert((char *) work_ + ocp_nlp_cost_external_workspace_calculate_size(config_, dims_, opts_) >= c_ptr);

    return;
}



/************************************************
 * functions
 ************************************************/

void ocp_nlp_cost_external_precompute(void *config_, void *dims_, void *model_, void *opts_, void *memory_, void *work_)
{
    return;
}



void ocp_nlp_cost_external_initialize(void *config_, void *dims_, void *model_, void *opts_,
                                      void *memory_, void *work_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    ocp_nlp_cost_external_memory *memory = memory_;

    // ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    int ns = dims->ns;

    blasfeo_dveccpsc(2*ns, model->scaling, &model->Z, 0, memory->Z, 0);

    return;
}



void ocp_nlp_cost_external_update_qp_matrices(void *config_, void *dims_, void *model_, void *opts_,
                                              void *memory_, void *work_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    ocp_nlp_cost_external_opts *opts = opts_;
    ocp_nlp_cost_external_memory *memory = memory_;
    ocp_nlp_cost_external_workspace *work = work_;

    ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    int nx = dims->nx;
    int nz = dims->nz;
    int nu = dims->nu;
    int ns = dims->ns;

    /* specify input types and pointers for external cost function */
    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[5];
    void *ext_fun_out[5];

    // INPUT
    struct blasfeo_dvec_args u_in;  // input u
    u_in.x = memory->ux;
    u_in.xi = 0;
    struct blasfeo_dvec_args x_in;  // input x
    x_in.x = memory->ux;
    x_in.xi = nu;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;
    ext_fun_type_in[2] = BLASFEO_DVEC;
    ext_fun_in[2] = memory->z_alg;

    // OUTPUT
    ext_fun_type_out[0] = COLMAJ;
    ext_fun_out[0] = &memory->fun;  // fun: scalar

    ext_fun_type_out[1] = BLASFEO_DVEC;
    ext_fun_out[1] = &work->tmp_nunxnz;  // tmp_nunxnz: nu+nx+nz

    if (opts->use_numerical_hessian > 0)
    {
        // evaluate external function
        model->ext_cost_fun_jac->evaluate(model->ext_cost_fun_jac, ext_fun_type_in,
                                            ext_fun_in, ext_fun_type_out, ext_fun_out);
        // custom hessian
        if (opts->add_hess_contribution)
        {
            blasfeo_dgead(nx+nu, nx+nu, model->scaling, &model->numerical_hessian, 0, 0, memory->RSQrq, 0, 0);
        }
        else
        {
            blasfeo_dgecpsc(nx+nu, nx+nu, model->scaling, &model->numerical_hessian, 0, 0, memory->RSQrq, 0, 0);
        }
    }
    else
    {
        // additional output
        ext_fun_type_out[2] = BLASFEO_DMAT;
        ext_fun_out[2] = &work->tmp_nunx_nunx;   // hess: (nu+nx) * (nu+nx)
        ext_fun_type_out[3] = BLASFEO_DMAT;
        ext_fun_out[3] = &work->tmp_nz_nz;       // hess_z: nz x nz
        ext_fun_type_out[4] = BLASFEO_DMAT;
        ext_fun_out[4] = &work->tmp_nz_nunx;    // hess_z_nunx: nz x nu+nx

        // evaluate external function
        model->ext_cost_fun_jac_hess->evaluate(model->ext_cost_fun_jac_hess, ext_fun_type_in,
                                            ext_fun_in, ext_fun_type_out, ext_fun_out);

        // hessian contribution from xu with scaling
        if (opts->add_hess_contribution)
        {
            // add to RSQrq
            blasfeo_dgead(nx+nu, nx+nu, model->scaling, &work->tmp_nunx_nunx, 0, 0, memory->RSQrq, 0, 0);
        }
        else
        {
            // copy to RSQrq
            blasfeo_dgecpsc(nx+nu, nx+nu, model->scaling, &work->tmp_nunx_nunx, 0, 0, memory->RSQrq, 0, 0);
        }

        if (nz > 0)
        {
            // NOTE: we compute the Hessian as follows:
            // H = d2l_dxu2 + dz_dux.T * d2l_dz2 * dz_dux + d2l_dux_dz * dz_dux + (d2l_dux_dz * dz_dux).T
            // the term d2z_dux2 is dropped!

            // compute and add cross terms (NOTE: only lower triangular is computed)
            blasfeo_dsyr2k_ln(nu+nx, nz, model->scaling, memory->dzdux_tran, 0, 0, &work->tmp_nz_nunx, 0, 0, 1., memory->RSQrq, 0, 0, memory->RSQrq, 0, 0);

            // hessian contribution from z
            blasfeo_dgemm_nt(nz, nu+nx, nz, 1., &work->tmp_nz_nz, 0, 0, memory->dzdux_tran, 0, 0, 0.0, &work->tmp_nz_nunx, 0, 0, &work->tmp_nz_nunx, 0, 0);
            blasfeo_dgemm_nn(nu+nx, nu+nx, nz, model->scaling, memory->dzdux_tran, 0, 0, &work->tmp_nz_nunx, 0, 0, 1., memory->RSQrq, 0, 0, memory->RSQrq, 0, 0);
        }
    }

    // gradient
    blasfeo_dveccp(nu+nx, &work->tmp_nunxnz, 0, &memory->grad, 0);
    if (nz > 0)
    {
        blasfeo_dgemv_n(nu+nx, nz, 1.0, memory->dzdux_tran, 0, 0, &work->tmp_nunxnz, nu+nx, 1., &memory->grad, 0, &memory->grad, 0);
    }

    // slack update gradient
    // grad_s (z_QP) = z_NLP + Z_NLP * slack
    blasfeo_dveccp(2*ns, &model->z, 0, &memory->grad, nu+nx);
    blasfeo_dvecmulacc(2*ns, &model->Z, 0, memory->ux, nu+nx, &memory->grad, nu+nx);

    // slack update function value
    // tmp_2ns = 2 * z + Z .* slack
    blasfeo_dveccpsc(2*ns, 2.0, &model->z, 0, &work->tmp_2ns, 0);
    blasfeo_dvecmulacc(2*ns, &model->Z, 0, memory->ux, nu+nx, &work->tmp_2ns, 0);
    // fun += .5 * (tmp_2ns .* slack)
    memory->fun += 0.5 * blasfeo_ddot(2*ns, &work->tmp_2ns, 0, memory->ux, nu+nx);

    // scale
    if (model->scaling!=1.0)
    {
        blasfeo_dvecsc(nu+nx+2*ns, model->scaling, &memory->grad, 0);
        memory->fun *= model->scaling;
    }

    // blasfeo_print_dmat(nu+nx, nu+nx, memory->RSQrq, 0, 0);
    // blasfeo_print_tran_dvec(2*ns, memory->Z, 0);
    // blasfeo_print_tran_dvec(nu+nx+2*ns, &memory->grad, 0);

    return;
}



void ocp_nlp_cost_external_compute_gradient(void *config_, void *dims_, void *model_, void *opts_,
                                 void *memory_, void *work_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    // ocp_nlp_cost_external_opts *opts = opts_;
    ocp_nlp_cost_external_memory *memory = memory_;
    ocp_nlp_cost_external_workspace *work = work_;

    ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    int nx = dims->nx;
    int nz = dims->nz;
    int nu = dims->nu;
    int ns = dims->ns;

    /* specify input types and pointers for external cost function */
    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[2];
    void *ext_fun_out[2];

    // INPUT
    struct blasfeo_dvec_args u_in;  // input u
    u_in.x = memory->ux;
    u_in.xi = 0;
    struct blasfeo_dvec_args x_in;  // input x
    x_in.x = memory->ux;
    x_in.xi = nu;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;
    ext_fun_type_in[2] = BLASFEO_DVEC;
    ext_fun_in[2] = memory->z_alg;

    // OUTPUT
    ext_fun_type_out[0] = COLMAJ;
    ext_fun_out[0] = &memory->fun;  // fun: scalar

    ext_fun_type_out[1] = BLASFEO_DVEC;
    ext_fun_out[1] = &work->tmp_nunxnz;  // tmp_nunxnz: nu+nx+nz

    // evaluate external function
    model->ext_cost_fun_jac->evaluate(model->ext_cost_fun_jac, ext_fun_type_in,
                                        ext_fun_in, ext_fun_type_out, ext_fun_out);

    // gradient
    blasfeo_dveccp(nu+nx, &work->tmp_nunxnz, 0, &memory->grad, 0);
    if (nz > 0)
    {
        blasfeo_dgemv_n(nu+nx, nz, 1.0, memory->dzdux_tran, 0, 0, &work->tmp_nunxnz, nu+nx, 1., &memory->grad, 0, &memory->grad, 0);
    }

    // slack update gradient
    blasfeo_dveccp(2*ns, &model->z, 0, &memory->grad, nu+nx);
    blasfeo_dvecmulacc(2*ns, &model->Z, 0, memory->ux, nu+nx, &memory->grad, nu+nx);

    // slack update function value
    // tmp_2ns = 2 * z + Z .* slack
    // blasfeo_dveccpsc(2*ns, 2.0, &model->z, 0, &work->tmp_2ns, 0);
    // blasfeo_dvecmulacc(2*ns, &model->Z, 0, memory->ux, nu+nx, &work->tmp_2ns, 0);
    // fun += .5 * (tmp_2ns .* slack)
    // memory->fun += 0.5 * blasfeo_ddot(2*ns, &work->tmp_2ns, 0, memory->ux, nu+nx);

    // scale
    if (model->scaling!=1.0)
    {
        blasfeo_dvecsc(nu+nx+2*ns, model->scaling, &memory->grad, 0);
        // memory->fun *= model->scaling;
    }
}



void ocp_nlp_cost_external_compute_fun(void *config_, void *dims_, void *model_,
                                       void *opts_, void *memory_, void *work_)
{

    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    // ocp_nlp_cost_external_opts *opts = opts_;
    ocp_nlp_cost_external_memory *memory = memory_;
    ocp_nlp_cost_external_workspace *work = work_;

    ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    struct blasfeo_dvec *ux = memory->ux;

    int nx = dims->nx;
    int nu = dims->nu;
    int ns = dims->ns;

    /* specify input types and pointers for external cost function */
    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[1];
    void *ext_fun_out[1];

    // INPUT
    struct blasfeo_dvec_args u_in;  // input u
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args x_in;  // input x
    x_in.x = ux;
    x_in.xi = nu;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;
    ext_fun_type_in[2] = BLASFEO_DVEC;
    ext_fun_in[2] = memory->z_alg;
    // OUTPUT
    ext_fun_type_out[0] = COLMAJ;
    ext_fun_out[0] = &memory->fun;  // function: scalar

    // evaluate external function
    if (model->ext_cost_fun == 0)
    {
        printf("ocp_nlp_cost_external_compute_fun: ext_cost_fun is not provided. Exiting.\n");
        exit(1);
    }
    model->ext_cost_fun->evaluate(model->ext_cost_fun, ext_fun_type_in, ext_fun_in,
                                  ext_fun_type_out, ext_fun_out);

    // slack update function value
    blasfeo_dveccpsc(2*ns, 2.0, &model->z, 0, &work->tmp_2ns, 0);
    blasfeo_dvecmulacc(2*ns, &model->Z, 0, ux, nu+nx, &work->tmp_2ns, 0);
    memory->fun += 0.5 * blasfeo_ddot(2*ns, &work->tmp_2ns, 0, ux, nu+nx);

    // scale
    if(model->scaling!=1.0)
    {
        memory->fun *= model->scaling;
    }

    return;
}

void ocp_nlp_cost_external_compute_jac_p(void *config_, void *dims_, void *model_,
                                       void *opts_, void *memory_, void *work_)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    ocp_nlp_cost_external_memory *memory = memory_;
    ocp_nlp_cost_external_workspace *work = work_;

    ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    struct blasfeo_dvec *ux = memory->ux;

    int nu = dims->nu;
    int nx = dims->nx;
    // int nz = dims->nz;
    int np_global = dims->np_global;

    /* specify input types and pointers for external cost function */
    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[1];
    void *ext_fun_out[1];

    // INPUT
    struct blasfeo_dvec_args u_in;  // input u
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args x_in;  // input x
    x_in.x = ux;
    x_in.xi = nu;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;
    ext_fun_type_in[2] = BLASFEO_DVEC;
    ext_fun_in[2] = memory->z_alg;

    // OUTPUT
    ext_fun_type_out[0] = BLASFEO_DMAT;
    ext_fun_out[0] = &work->cost_grad_params_jac;

    // evaluate external function
    if (model->ext_cost_hess_xu_p == 0)
    {
        printf("ocp_nlp_cost_external_compute_jac_p: ext_cost_hess_xu_p is not provided. Exiting.\n");
        exit(1);
    }
    model->ext_cost_hess_xu_p->evaluate(model->ext_cost_hess_xu_p, ext_fun_type_in, ext_fun_in,
                                  ext_fun_type_out, ext_fun_out);

    // add contribution to stationarity jacobian:
    // jac_lag_stat_p_global += scaling * cost_grad_params_jac
    blasfeo_dgead(nu+nx, np_global, model->scaling, &work->cost_grad_params_jac, 0, 0, memory->jac_lag_stat_p_global, 0, 0);

    return;
}

void ocp_nlp_cost_external_eval_grad_p(void *config_, void *dims_, void *model_, void *opts_, void *memory_, void *work_, struct blasfeo_dvec *out)
{
    ocp_nlp_cost_external_dims *dims = dims_;
    ocp_nlp_cost_external_model *model = model_;
    ocp_nlp_cost_external_memory *memory = memory_;

    ocp_nlp_cost_external_cast_workspace(config_, dims, opts_, work_);

    struct blasfeo_dvec *ux = memory->ux;

    int nu = dims->nu;
    // int nx = dims->nx;
    // int nz = dims->nz;
    int np_global = dims->np_global;

    /* specify input types and pointers for external cost function */
    ext_fun_arg_t ext_fun_type_in[3];
    void *ext_fun_in[3];
    ext_fun_arg_t ext_fun_type_out[1];
    void *ext_fun_out[1];

    // INPUT
    struct blasfeo_dvec_args u_in;  // input u
    u_in.x = ux;
    u_in.xi = 0;

    struct blasfeo_dvec_args x_in;  // input x
    x_in.x = ux;
    x_in.xi = nu;

    ext_fun_type_in[0] = BLASFEO_DVEC_ARGS;
    ext_fun_in[0] = &x_in;
    ext_fun_type_in[1] = BLASFEO_DVEC_ARGS;
    ext_fun_in[1] = &u_in;
    ext_fun_type_in[2] = BLASFEO_DVEC;
    ext_fun_in[2] = memory->z_alg;

    // OUTPUT
    ext_fun_type_out[0] = BLASFEO_DVEC;
    ext_fun_out[0] = out;

    // evaluate external function
    model->ext_cost_grad_p->evaluate(model->ext_cost_grad_p, ext_fun_type_in, ext_fun_in,
                                  ext_fun_type_out, ext_fun_out);

    // scale
    if(model->scaling != 1.0)
    {
        blasfeo_dvecsc(np_global, model->scaling, out, 0);
    }

    return;
}


size_t ocp_nlp_cost_external_get_external_fun_workspace_requirement(void *config_, void *dims_, void *opts_, void *model_)
{
    ocp_nlp_cost_external_model *model = model_;

    size_t size = 0;
    size_t tmp_size;

    tmp_size = external_function_get_workspace_requirement_if_defined(model->ext_cost_fun);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->ext_cost_fun_jac);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->ext_cost_fun_jac_hess);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->ext_cost_grad_p);
    size = size > tmp_size ? size : tmp_size;
    tmp_size = external_function_get_workspace_requirement_if_defined(model->ext_cost_hess_xu_p);
    size = size > tmp_size ? size : tmp_size;

    return size;
}


void ocp_nlp_cost_external_set_external_fun_workspaces(void *config_, void *dims_, void *opts_, void *model_, void *workspace_)
{
    ocp_nlp_cost_external_model *model = model_;
    external_function_set_fun_workspace_if_defined(model->ext_cost_fun, workspace_);
    external_function_set_fun_workspace_if_defined(model->ext_cost_fun_jac, workspace_);
    external_function_set_fun_workspace_if_defined(model->ext_cost_fun_jac_hess, workspace_);
    external_function_set_fun_workspace_if_defined(model->ext_cost_grad_p, workspace_);
    external_function_set_fun_workspace_if_defined(model->ext_cost_hess_xu_p, workspace_);
}



/* config */

void ocp_nlp_cost_external_config_initialize_default(void *config_, int stage)
{
    ocp_nlp_cost_config *config = config_;

    config->dims_calculate_size = &ocp_nlp_cost_external_dims_calculate_size;
    config->dims_assign = &ocp_nlp_cost_external_dims_assign;
    config->dims_set = &ocp_nlp_cost_external_dims_set;
    config->dims_get = &ocp_nlp_cost_external_dims_get;
    config->model_calculate_size = &ocp_nlp_cost_external_model_calculate_size;
    config->model_assign = &ocp_nlp_cost_external_model_assign;
    config->model_set = &ocp_nlp_cost_external_model_set;
    config->model_get = &ocp_nlp_cost_external_model_get;
    config->model_get_scaling_ptr = &ocp_nlp_cost_external_model_get_scaling_ptr;
    config->opts_calculate_size = &ocp_nlp_cost_external_opts_calculate_size;
    config->opts_assign = &ocp_nlp_cost_external_opts_assign;
    config->opts_initialize_default = &ocp_nlp_cost_external_opts_initialize_default;
    config->opts_update = &ocp_nlp_cost_external_opts_update;
    config->opts_set = &ocp_nlp_cost_external_opts_set;
    config->opts_get_add_hess_contribution_ptr = &ocp_nlp_cost_external_opts_get_add_hess_contribution_ptr;
    config->memory_calculate_size = &ocp_nlp_cost_external_memory_calculate_size;
    config->memory_assign = &ocp_nlp_cost_external_memory_assign;
    config->memory_get_fun_ptr = &ocp_nlp_cost_external_memory_get_fun_ptr;
    config->memory_get_grad_ptr = &ocp_nlp_cost_external_memory_get_grad_ptr;
    config->memory_set_ux_ptr = &ocp_nlp_cost_external_memory_set_ux_ptr;
    config->memory_set_z_alg_ptr = &ocp_nlp_cost_external_memory_set_z_alg_ptr;
    config->memory_set_dzdux_tran_ptr = &ocp_nlp_cost_external_memory_set_dzdux_tran_ptr;
    config->memory_set_RSQrq_ptr = &ocp_nlp_cost_external_memory_set_RSQrq_ptr;
    config->memory_set_Z_ptr = &ocp_nlp_cost_external_memory_set_Z_ptr;
    config->memory_set_jac_lag_stat_p_global_ptr = &ocp_nlp_cost_external_memory_set_jac_lag_stat_p_global_ptr;
    config->workspace_calculate_size = &ocp_nlp_cost_external_workspace_calculate_size;
    config->get_external_fun_workspace_requirement = &ocp_nlp_cost_external_get_external_fun_workspace_requirement;
    config->set_external_fun_workspaces = &ocp_nlp_cost_external_set_external_fun_workspaces;
    config->initialize = &ocp_nlp_cost_external_initialize;
    config->update_qp_matrices = &ocp_nlp_cost_external_update_qp_matrices;
    config->compute_fun = &ocp_nlp_cost_external_compute_fun;
    config->compute_jac_p = &ocp_nlp_cost_external_compute_jac_p;
    config->compute_gradient = &ocp_nlp_cost_external_compute_gradient;
    config->eval_grad_p = &ocp_nlp_cost_external_eval_grad_p;
    config->config_initialize_default = &ocp_nlp_cost_external_config_initialize_default;
    config->precompute = &ocp_nlp_cost_external_precompute;
    config->stage = stage;

    return;
}
