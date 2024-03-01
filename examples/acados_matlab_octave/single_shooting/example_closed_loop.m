%
% Copyright (c) The acados authors.
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;

%

%% test of native matlab interface
clear all; clc; close all;
check_acados_requirements()

%% create the controller
h = 1/14;   % [s] sampling time
N = 10;     % [-] number of prediction steps
T = N*h;    % [s] prediction horizon length

model = quadcopter(N,h);    % system model
nu = size(model.B,2) / N;   % input dimension
ny = size(model.B,1) / N;   % output dimension 
nx = size(model.A,2);       % state dimension
C = model.C;                % output matrix

% ocp model
ocp_model = acados_ocp_model();
ocp_model.set('name', 'quad_mpc');
ocp_model.set('T', h);
ocp_model.set('sym_x', model.sym_x);
ocp_model.set('sym_u', model.sym_u);
ocp_model.set('sym_p', model.sym_p);

% cost
cost_type = 'auto';  % auto, ext_cost, linear_ls
ocp_model.set('cost_type_0', cost_type);
ocp_model.set('cost_type', cost_type);
ocp_model.set('cost_type_e', cost_type);
if strcmp(cost_type,'linear_ls')
    ocp_model.set('cost_Vu_0', model.Vu_0);
    ocp_model.set('cost_Vx_0', model.Vx_0);
    ocp_model.set('cost_W_0', model.W_0);
    ocp_model.set('cost_y_ref_0', model.y_ref_0);

    ocp_model.set('cost_Vu', model.Vu);
    ocp_model.set('cost_Vx', model.Vx);
    ocp_model.set('cost_W', model.W);
    ocp_model.set('cost_y_ref', model.y_ref);

    ocp_model.set('cost_Vx_e', model.Vx_e);    
    ocp_model.set('cost_W_e', model.W_e);    
    ocp_model.set('cost_y_ref_e', model.y_ref_e);
else
    ocp_model.set('cost_expr_ext_cost_0', model.cost_expr_ext_cost_0);
    ocp_model.set('cost_expr_ext_cost', model.cost_expr_ext_cost);
    ocp_model.set('cost_expr_ext_cost_e', model.cost_expr_ext_cost_e);
end

% dynamics
ocp_model.set('dyn_type', 'discrete');
ocp_model.set('dyn_expr_phi', model.dyn_expr_phi);

% constraints (initial state, intermediate states, terminal state, inputs)
ocp_model.set('constr_Jbx_e', model.Jbx_e);
ocp_model.set('constr_lbx_e', model.lbx_e);
ocp_model.set('constr_ubx_e', model.ubx_e);
ocp_model.set('constr_Jbu', model.Jbu);
ocp_model.set('constr_lbu', model.lbu);
ocp_model.set('constr_ubu', model.ubu);

% ocp options
ocp_opts = acados_ocp_opts();
ocp_opts.set('param_scheme_N', 1);
ocp_opts.set('sim_method', 'discrete');
ocp_opts.set('qp_solver','partial_condensing_hpipm');
ocp_opts.set('parameter_values', zeros(nx+N*ny,1));  % initialize to zero, change later

ocp = acados_ocp(ocp_model, ocp_opts);

%% simulate the closed-loop system
x0 = zeros(ny,1);  % initial state
yr = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];  % output reference
nsim = 15;  % number of simulation steps
Y = nan(ny,nsim+1);  % output log
Y(:,1) = C*x0;  % first entry is the initial output
U = nan(nu,nsim);  % control input log
solve_time_log = nan(1,nsim);  % for solver performance evaluation

for i = 1 : nsim
    % set the parameters
    if strcmp(cost_type,'linear_ls')
        ocp.set('cost_y_ref_e', repmat(yr,N,1));  % for the terminal cost
    end
    ocp.set('p', [x0; repmat(yr,N,1)]);
    
    % solve the ocp
    ocp.solve();

    % check the solver output
    if ocp.get('status') ~= 0
        warning(['acados ocp solver failed with status ',num2str(status)]);
    end

    % apply the first control input to the plant, simulate one step
    Ctrl = ocp.get('u', 0);  % stacked control inputs
    ctrl = Ctrl(1:nu);  % get the first control input
    x0 = model.Ad*x0 + model.Bd*ctrl;  % simulation

    % log the data
    Y(:,i+1) = C*x0;
    U(:,i) = ctrl;
    solve_time_log(i) = ocp.get('time_tot');
end

disp([newline,'Average solve time: ',num2str(1e3*mean(solve_time_log)),' ms'])

%% plot the results
figure
subplot(2,1,1)
plot(Y(3,:))
hold on; yline(yr(3),'r--')
ylim([-0.1 1.1])
xlim([1 nsim])
ylabel('$x_3$')
subplot(2,1,2)
stairs(U')
ylim([-1.1 2])
xlim([1 nsim])
ylabel('$u$')
xlabel('step')