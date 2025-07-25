#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#


# OCTAVE
if(ACADOS_OCTAVE)
    # new minimal examples
    add_test(NAME octave_new_ocp
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/getting_started
        octave --no-gui --no-window-system ./minimal_example_ocp.m)
    add_test(NAME octave_new_sim
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/getting_started
        octave --no-gui --no-window-system ./minimal_example_sim.m)
    # dim check test
    add_test(NAME octave_test_dim_check
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_dim_check.m)

    # SIM
    add_test(NAME octave_test_sim_forw
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_sim_forw.m)

    add_test(NAME octave_test_sim_adj
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_sim_adj.m)

    add_test(NAME octave_test_sim_hess
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_sim_hess.m)

    add_test(NAME octave_test_sim_dae
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_sim_dae.m)

    # OCP
    add_test(NAME octave_test_ocp_pendulum
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_ocp_pendulum.m)
    add_test(NAME octave_test_ocp_pendulum_code_reuse
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./test_code_reuse.m)
    add_test(NAME octave_test_ocp_wtnx6
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_ocp_wtnx6.m)
    add_test(NAME octave_test_ocp_mass_spring
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./run_test_ocp_mass_spring.m)
    add_test(NAME octave_test_ocp_simple_dae
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/simple_dae_model
        octave --no-gui --no-window-system ./example_ocp.m)

    # AS-RTI closed loop example
    add_test(NAME octave_closed_loop_as_rti_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/pendulum_on_cart_model
        octave --no-gui --no-window-system ./closed_loop_as_rti.m)

    add_test(NAME octave_test_mhe_lorentz
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/lorentz
        octave --no-gui --no-window-system ./example_mhe.m)

    # OSQP
    if(ACADOS_WITH_OSQP)
        add_test(NAME octave_test_OSQP
            COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
            octave --no-gui --no-window-system ./test_ocp_OSQP.m)
    endif()

    # qpDUNES
    if(ACADOS_WITH_QPDUNES)
    add_test(NAME octave_test_qpDUNES
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./test_ocp_qpdunes.m)
    endif()
    #
    add_test(NAME octave_test_target_selector
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/test
        octave --no-gui --no-window-system ./test_target_selector.m)
    add_test(NAME octave_test_generic_ext_cost_disc_dyn
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/generic_dyn_disc
        octave --no-gui --no-window-system ./run_generic_disc_dyn_example.m)

    # MOCP
    add_test(NAME octave_mocp_transition_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_matlab_octave/mocp_transition_example
        octave --no-gui --no-window-system ./main_multiphase_ocp.m)

    # Force serial execution for tests in the same directory
    # Directory test
    set_tests_properties(octave_new_ocp PROPERTIES DEPENDS octave_new_sim)

    set_tests_properties(octave_test_target_selector PROPERTIES DEPENDS octave_test_generic_ext_cost_disc_dyn)
    set_tests_properties(octave_test_generic_ext_cost_disc_dyn PROPERTIES DEPENDS octave_test_dim_check)
    set_tests_properties(octave_test_dim_check PROPERTIES DEPENDS octave_test_sim_forw)
    set_tests_properties(octave_test_sim_forw PROPERTIES DEPENDS octave_test_sim_adj)
    set_tests_properties(octave_test_sim_adj PROPERTIES DEPENDS octave_test_sim_hess)
    set_tests_properties(octave_test_sim_hess PROPERTIES DEPENDS octave_test_sim_dae)
    set_tests_properties(octave_test_sim_dae PROPERTIES DEPENDS octave_test_ocp_pendulum)
    set_tests_properties(octave_test_ocp_pendulum PROPERTIES DEPENDS octave_test_ocp_wtnx6)
    set_tests_properties(octave_test_ocp_wtnx6 PROPERTIES DEPENDS octave_test_ocp_mass_spring)
    set_tests_properties(octave_test_ocp_mass_spring PROPERTIES DEPENDS octave_test_ocp_simple_dae)
    set_tests_properties(octave_test_ocp_simple_dae PROPERTIES DEPENDS octave_test_target_selector)
    set_tests_properties(octave_test_target_selector PROPERTIES DEPENDS octave_test_ocp_pendulum_code_reuse)
    if(ACADOS_WITH_OSQP)
        set_tests_properties(octave_test_ocp_pendulum_code_reuse PROPERTIES DEPENDS octave_test_OSQP)
    endif()
    if(ACADOS_WITH_QPDUNES)
        set_tests_properties(octave_test_ocp_pendulum_code_reuse PROPERTIES DEPENDS octave_test_qpDUNES)
        set_tests_properties(octave_test_OSQP PROPERTIES DEPENDS octave_test_qpDUNES)
    endif()
endif()


### PYTHON ###
if(ACADOS_PYTHON)
    # Minimal examples
    add_test(NAME python_pendulum_closed_loop_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/getting_started
        python minimal_example_closed_loop.py)
    add_test(NAME python_pendulum_sim_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/getting_started
        python minimal_example_sim.py)
    add_test(NAME python_pendulum_ocp_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/getting_started
        python minimal_example_ocp.py)

    add_test(NAME python_test_generic_impl_dyn
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/generic_impl_dyn
        python minimal_example_ocp_generic_impl_dyn.py)
    add_test(NAME python_test_reset
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python reset_test.py)
    add_test(NAME python_test_reset_timing
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/timing_example
        python reset_timing.py)
    add_test(NAME python_test_cython_vs_ctypes
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_cython_ctypes.py)
    add_test(NAME python_test_ocp
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python main_test.py)
    add_test(NAME python_test_detect_constraints
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_detect_constraints.py)

    add_test(NAME python_test_cost_integration_euler
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_cost_integration_euler.py)

    add_test(NAME python_test_cost_integration_value
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_cost_integration_value.py)

    add_test(NAME python_pendulum_ocp_example_reuse_code
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python minimal_example_ocp_reuse_code.py)

    # add_test(NAME python_solution_sensitivities_and_exact_hess
    #     COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/solution_sensitivities
    #     python test_solution_sens_and_exact_hess.py)

    # add_test(NAME python_value_function_gradient
    #     COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/solution_sensitivities
    #     python value_gradient_example.py)

    # add_test(NAME python_policy_gradient
    #     COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/solution_sensitivities
    #     python policy_gradient_example.py)

    add_test(NAME python_time_varying_irk
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/time_varying
        python test_time_varying_irk.py)

    add_test(NAME test_polynomial_controls_and_penalties
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/time_varying
        python test_polynomial_controls_and_penalties.py)

    add_test(NAME py_mocp_qp_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/time_varying
        python test_mocp_qp.py)


# Tests for DDP
add_test(NAME py_ddp_solve_qp
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/unconstrained_ocps/linear_dynamics_qp_ocp
        python acados_unconstrained_QP.py)

add_test(NAME py_ddp_chen_allgoewer
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/unconstrained_ocps/chen_allgoewer_unconstrained_ocp
        python chen_allgoewer_ocp.py)

add_test(NAME py_hour_glass_p2p_motion
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/unconstrained_ocps/hour_glass_p2p_motion
        python hour_glass_time_optimal_p2p_motion.py)

add_test(NAME py_ddp_rockit_hello_world
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/unconstrained_ocps/rockit_hello_world
        python rockit_hello_world_ocp.py)

# Tests for SQP_WITH_FEASIBLE_QP
add_test(NAME py_sqp_wfqp_linear_problem_obstacle_avoidance
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/linear_mass_model
        python sqp_wfqp_test.py)

add_test(NAME py_sqp_wfqp_inconsistent_linearization
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/inconsistent_qp_linearization
        python inconsistent_qp_linearization_test.py)

add_test(NAME py_sqp_wfqp_problem_hs015
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/hock_schittkowsky
        python hs015_test.py)

add_test(NAME py_sqp_wfqp_problem_hs016
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/hock_schittkowsky
        python hs016_test.py)

add_test(NAME py_sqp_wfqp_problem_hs074_constraint_scaling
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/hock_schittkowsky
        python hs074_constraint_scaling.py)

add_test(NAME py_hs099_tol_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/hock_schittkowsky
        python hs099.py)


# CMake test
add_test(NAME python_pendulum_ocp_example_cmake
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python minimal_example_ocp_cmake.py)
    add_test(NAME python_nonuniform_discretization_ocp_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python nonuniform_discretization_example.py)
    add_test(NAME python_rti_loop_ocp_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python example_sqp_rti_loop.py)
    # Python Simulink
    add_test(NAME python_render_simulink_wrapper
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python simulink_example.py)
    add_test(NAME python_constraints_expression_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python ocp_example_h_init_contraints.py)

    add_test(NAME python_chain_ocp
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/chain_mass
        python main.py)

    add_test(NAME python_chain_sim
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/chain_mass
        python minimal_example_sim.py)

    # Maratos test problem with different globalization options
    add_test(NAME python_maratos_test_problem_globalization
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/non_ocp_nlp
        python maratos_test_problem.py)

    add_test(NAME python_test_adaptive_reg
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/non_ocp_nlp
        python adaptive_eps_reg_test.py)

    add_test(NAME py_qp_scaling_non_ocp
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/non_ocp_nlp
        python qpscaling_test.py)

    # Convex test problem where full step SQP does not converge, but globalized SQP does
    add_test(NAME python_convex_test_problem_globalization
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/convex_problem_globalization_needed
        python convex_problem_globalization_necessary.py)

    add_test(NAME python_max_iter_termination_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/max_iter_test
        python max_iter_test.py)

    # Simple OCP with Maratos effect
    add_test(NAME python_OCP_maratos_test_problem_globalization
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/linear_mass_model
        python linear_mass_test_problem.py)

    add_test(NAME py_qpscaling_slacked
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/linear_mass_model
        python test_qpscaling_slacked.py)

    # Armijo test problem
    add_test(NAME python_armijo_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python armijo_test.py)

    # Test NaN in globalization
    add_test(NAME python_test_nan_globalization
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_nan_globalization.py)

    # Multiphase nonlinear constraint test problem
    add_test(NAME python_multiphase_nonlinear_constraints
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/multiphase_nonlinear_constraints
        python main.py)

    # OSQP test
    if(ACADOS_WITH_OSQP)
    add_test(NAME python_OSQP_test
            COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
            python test_osqp.py)
    endif()

    # MHE examples
    add_test(NAME python_pendulum_mhe_example_minimal
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/mhe
        python minimal_example_mhe.py)
    add_test(NAME python_pendulum_mhe_example_noisy_param
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/mhe
        python minimal_example_mhe_with_noisy_param.py)
    add_test(NAME python_pendulum_mhe_example_param
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/mhe
        python minimal_example_mhe_with_param.py)
    add_test(NAME python_pendulum_mhe_ocp_closed_loop_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/mhe
        python closed_loop_mhe_ocp.py)

    add_test(NAME python_custom_update_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/custom_update
        python example_custom_rti_loop.py)

    add_test(NAME python_as_rti_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/as_rti
        python as_rti_closed_loop_example.py)

    add_test(NAME python_fast_zoro_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/zoRO_example
        python pendulum_on_cart/minimal_example_zoro.py)

    add_test(NAME python_zoro_diff_drive_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/zoRO_example
        python diff_drive/main.py)

    add_test(NAME python_convex_ocp_with_onesided_constraints
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/convex_ocp_with_onesided_constraints
        python main_convex_onesided.py)

    # casadi_examples
    add_test(NAME python_casadi_get_set_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/casadi_tests
        python test_casadi_get_set.py)
    add_test(NAME test_casadi_parametric
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/casadi_tests
        python test_casadi_parametric.py)
    add_test(NAME test_casadi_p_in_constraint_and_cost
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/casadi_tests
        python test_casadi_p_in_constraint_and_cost.py)
    add_test(NAME test_casadi_constraint
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/casadi_tests
        python test_casadi_constraint.py)
    add_test(NAME test_casadi_slack_in_h
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/casadi_tests
        python test_casadi_slack_in_h.py)

    # Sim
    add_test(NAME python_pendulum_ext_sim_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/sim
        python extensive_example_sim.py
    )

    add_test(NAME cython_pendulum_closed_loop_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart
        python cython_example_closed_loop.py)
    add_test(NAME py_time_optimal_cython_ctypes
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/crane
        python time_optimal_example.py)
    add_test(NAME pendulum_optimal_value_gradient
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python example_optimal_value_derivative.py)

    # example_ocp_dynamics_formulations all versions
    add_test(NAME python_pendulum_ocp_IRK
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python example_ocp_dynamics_formulations.py --INTEGRATOR_TYPE=IRK)

    if(ACADOS_WITH_QPDUNES)
        add_test(NAME python_pendulum_ocp_ERK_qpDUNES
            COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
            python example_ocp_dynamics_formulations.py --INTEGRATOR_TYPE=ERK --QP_SOLVER=PARTIAL_CONDENSING_QPDUNES)
    endif()

    add_test(NAME python_pendulum_ocp_GNSF
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python example_ocp_dynamics_formulations.py --INTEGRATOR_TYPE=GNSF)

    add_test(NAME py_qp_scaling_time_opt_swingup
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python time_optimal_swing_up.py)

    # CMake and solver=DISCRETE test
    add_test(NAME python_example_ocp_dynamics_formulations_cmake
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pendulum_on_cart/ocp
        python example_ocp_dynamics_formulations.py --INTEGRATOR_TYPE DISCRETE --BUILD_SYSTEM cmake)

    add_test(NAME python_pendulum_soft_constraints_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python soft_constraint_test.py)
    add_test(NAME python_pendulum_parametric_nonlinear_constraint_h_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_parametric_nonlinear_constraint_h.py)
    add_test(NAME python_test_sim_dae
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python test_sim_dae.py)
    add_test(NAME python_sparse_param_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python sparse_param_test.py)
    add_test(NAME python_regularization_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python regularization_test.py)
    add_test(NAME python_one_sided_constraints_test
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/tests
        python one_sided_constraints_test.py)


    add_test(NAME python_pmsm_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/pmsm_example
        python main.py)
    add_test(NAME python_quadrotor_nav
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/quadrotor_nav
        python main.py)
    add_test(NAME python_race_cars
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/race_cars
        python main.py)
    add_test(NAME python_rsm_example
        COMMAND "${CMAKE_COMMAND}" -E chdir ${PROJECT_SOURCE_DIR}/examples/acados_python/rsm_example
        python main.py)

    # Force serial execution for all the below specified tests despite multiple CPU aviliablity (conflicting object file naming)
    # TODO use unique object file names to allow full parallization, and remove  forced serialization section below
    # Directory zoRO_example
    set_tests_properties(python_fast_zoro_example PROPERTIES DEPENDS python_zoro_diff_drive_example)

    # casadi_tests
    set_tests_properties(python_casadi_get_set_example PROPERTIES DEPENDS test_casadi_p_in_constraint_and_cost)
    set_tests_properties(test_casadi_p_in_constraint_and_cost PROPERTIES DEPENDS test_casadi_parametric)
    set_tests_properties(test_casadi_constraint PROPERTIES DEPENDS python_casadi_get_set_example)
    set_tests_properties(test_casadi_slack_in_h PROPERTIES DEPENDS test_casadi_constraint)

    # Directory getting_started
    set_tests_properties(python_pendulum_sim_example PROPERTIES DEPENDS python_pendulum_ocp_example)
    set_tests_properties(python_pendulum_closed_loop_example PROPERTIES DEPENDS python_pendulum_sim_example)

    # Directory non_ocp_nlp
    set_tests_properties(python_maratos_test_problem_globalization PROPERTIES DEPENDS python_test_adaptive_reg)
    set_tests_properties(python_test_adaptive_reg PROPERTIES DEPENDS py_qp_scaling_non_ocp)

    # Directory acados_python/tests
    set_tests_properties(python_test_cython_vs_ctypes PROPERTIES DEPENDS python_test_reset)
    set_tests_properties(python_test_reset PROPERTIES DEPENDS python_test_ocp)
    set_tests_properties(python_test_ocp PROPERTIES DEPENDS python_test_cost_integration_euler)
    set_tests_properties(python_test_cost_integration_euler PROPERTIES DEPENDS python_one_sided_constraints_test)
    set_tests_properties(python_one_sided_constraints_test PROPERTIES DEPENDS python_test_cost_integration_value)
    set_tests_properties(python_test_cost_integration_value PROPERTIES DEPENDS python_armijo_test)
    set_tests_properties(python_armijo_test PROPERTIES DEPENDS python_pendulum_soft_constraints_example)
    set_tests_properties(python_pendulum_soft_constraints_example PROPERTIES DEPENDS python_pendulum_parametric_nonlinear_constraint_h_test)
    set_tests_properties(python_pendulum_parametric_nonlinear_constraint_h_test PROPERTIES DEPENDS python_test_nan_globalization)
    set_tests_properties(python_test_nan_globalization PROPERTIES DEPENDS python_regularization_test)
    set_tests_properties(python_regularization_test PROPERTIES DEPENDS python_test_sim_dae)
    set_tests_properties(python_test_sim_dae PROPERTIES DEPENDS python_sparse_param_test)
    set_tests_properties(python_sparse_param_test PROPERTIES DEPENDS python_test_detect_constraints)

    if(ACADOS_WITH_OSQP)
        set_tests_properties(python_test_detect_constraints PROPERTIES DEPENDS python_OSQP_test)
    endif()

    # Directory acados_python/chain_mass
    set_tests_properties(python_chain_ocp PROPERTIES DEPENDS python_chain_sim)

    # Directory pendulum_on_cart/sim
    set_tests_properties(python_pendulum_ext_sim_example PROPERTIES DEPENDS python_pendulum_sim_example_cmake)

    # Directory pendulum_on_cart/mhe
    set_tests_properties(python_pendulum_mhe_example_minimal PROPERTIES DEPENDS python_pendulum_mhe_example_noisy_param)
    set_tests_properties(python_pendulum_mhe_example_noisy_param PROPERTIES DEPENDS python_pendulum_mhe_example_param)
    set_tests_properties(python_pendulum_mhe_example_param PROPERTIES DEPENDS python_pendulum_mhe_ocp_closed_loop_example)

    # # Directory  pendulum_on_cart/solution_sensitivities
    # set_tests_properties(python_solution_sensitivities_and_exact_hess PROPERTIES DEPENDS python_value_function_gradient)
    # set_tests_properties(python_value_function_gradient PROPERTIES DEPENDS python_policy_gradient)

    # Directory time_varying
    set_tests_properties(python_time_varying_irk PROPERTIES DEPENDS test_polynomial_controls_and_penalties)
    set_tests_properties(test_polynomial_controls_and_penalties PROPERTIES DEPENDS py_mocp_qp_test)

    # Directory linear_mass_model
    set_tests_properties(python_OCP_maratos_test_problem_globalization PROPERTIES DEPENDS py_qpscaling_slacked)

    # Directory pendulum_on_cart/ocp
    set_tests_properties(python_pendulum_ocp_example_reuse_code PROPERTIES DEPENDS python_nonuniform_discretization_ocp_example)
    set_tests_properties(python_nonuniform_discretization_ocp_example PROPERTIES DEPENDS python_rti_loop_ocp_example)
    set_tests_properties(python_rti_loop_ocp_example PROPERTIES DEPENDS python_render_simulink_wrapper)
    set_tests_properties(python_render_simulink_wrapper PROPERTIES DEPENDS python_constraints_expression_example)
    set_tests_properties(python_constraints_expression_example PROPERTIES DEPENDS pendulum_optimal_value_gradient)
    set_tests_properties(pendulum_optimal_value_gradient PROPERTIES DEPENDS python_pendulum_ocp_IRK)
    set_tests_properties(python_pendulum_ocp_IRK PROPERTIES DEPENDS python_pendulum_ocp_GNSF)
    set_tests_properties(python_pendulum_ocp_GNSF PROPERTIES DEPENDS python_example_ocp_dynamics_formulations_cmake)
    set_tests_properties(python_example_ocp_dynamics_formulations_cmake PROPERTIES DEPENDS py_qp_scaling_time_opt_swingup)

    if(ACADOS_WITH_QPDUNES)
    set_tests_properties(python_example_ocp_dynamics_formulations_cmake PROPERTIES DEPENDS python_pendulum_ocp_ERK_qpDUNES)
    endif()

endif()
