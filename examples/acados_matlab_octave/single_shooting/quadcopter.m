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

% this function implements the quadcopter model from https://osqp.org/docs/examples/mpc.html
% more infomation can be found here: https://github.com/orgs/osqp/discussions/558

function model = quadcopter(N,h)
% inputs: number of prediction steps, sampling time
import casadi.*

% system matrices
Ad = [1       0       0   0   0   0   0.1     0       0    0       0       0;
      0       1       0   0   0   0   0       0.1     0    0       0       0;
      0       0       1   0   0   0   0       0       0.1  0       0       0;
      0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
      0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
      0       0       0   0   0   1   0       0       0    0       0       0.0992;
      0       0       0   0   0   0   1       0       0    0       0       0;
      0       0       0   0   0   0   0       1       0    0       0       0;
      0       0       0   0   0   0   0       0       1    0       0       0;
      0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
      0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
      0       0       0   0   0   0   0       0       0    0       0       0.9846];
Bd = [0      -0.0726  0       0.0726;
     -0.0726  0       0.0726  0;
     -0.0152  0.0152 -0.0152  0.0152;
      0      -0.0006 -0.0000  0.0006;
      0.0006  0      -0.0006  0;
      0.0106  0.0106  0.0106  0.0106;
      0      -1.4512  0       1.4512;
     -1.4512  0       1.4512  0;
     -0.3049  0.3049 -0.3049  0.3049;
      0      -0.0236  0       0.0236;
      0.0236  0      -0.0236  0;
      0.2107  0.2107  0.2107  0.2107];
[nx, nu] = size(Bd);  % state and input dimensions
Cd = eye(nx);  % all states are measured
ny = size(Cd,1);

% dense form matrices, Y = A*x0 + B*U
[A,B] = dense_prediction_matrices(Ad,Bd,Cd,N);

% (unnamed) symbolic variables
Y = SX.sym('Y',ny*N,1);  % stacked output vector
U = SX.sym('U',nu*N,1);  % stacked input vector
x0 = SX.sym('x0',nx,1);  % initial state
Yr = SX.sym('Yr',ny*N,1);  % stacked output reference
sym_p = vertcat(x0,Yr);  % [x0; repmat(yr,N)]

% discrete system dynamics
dyn_expr_phi = A * x0 + B * U;

% cost matrices
Qd = diag([0 0 10 10 10 10 0 0 0 5 5 5]);  % state cost
Q = kron(eye(N),Qd);  % stacked state cost
Rd = 0.1*eye(4);  % input cost
R = kron(eye(N),Rd);  % stacked input cost

% generic cost formulation
cost_expr_ext_cost_e = (Y-Yr)'*Q*(Y-Yr);    % terminal cost (outputs only)
cost_expr_ext_cost_0 = U'*R*U;              % initial stage cost (inputs only)
% cost_expr_ext_cost_0 = 1/h * cost_expr_ext_cost_0;  % scale the stage cost to match the discrete formulation
% more info on discrete cost scaling: 
% https://discourse.acados.org/t/question-regarding-terminal-cost-in-discrete-time/1096

% linear least-squares cost formulation (alternative)
% initial cost (inputs only)
ny_0 = nu*N;  % number of outputs in the stage cost
Vu_0 = eye(ny_0);  % input-to-output matrix in the stage cost
Vx_0 = zeros(ny_0,ny*N);  % state-to-output matrix in the stage cost
W_0 = 2*R;  % weight matrix in the stage cost
y_ref_0 = zeros(ny_0, 1);  % output reference in the stage cost

% no stage cost

% terminal cost (outputs only)
ny_e = ny*N;  % number of outputs in the terminal cost
Vx_e = eye(ny_e);  % state-to-output matrix in the terminal cost
W_e = 2*Q;  % weight matrix in the terminal cost
y_ref_e = zeros(ny_e, 1);  % output reference in the terminal cost

% input constraints
u0 = 10.5916;  % steady-state input
Jbu = eye(nu);  % all inputs are constrained
lbu = [9.6; 9.6; 9.6; 9.6] - u0;  % input lower bounds
ubu = [13; 13; 13; 13] - u0;  % input upper bounds

% stacked input constraints
JbU = kron(eye(N),Jbu);
Lbu = repmat(lbu,N,1);
Ubu = repmat(ubu,N,1);

% state constraints on the first, second and sixth state
Jbx = zeros(3,nx);
Jbx(1,1) = 1;
Jbx(2,2) = 1;
Jbx(3,6) = 1;
infty = 1e6;  % to approximate one-sided constraints
lbx = [-pi/6; -pi/6; -1];  % state lower bounds
ubx = [ pi/6;  pi/6; infty];  % state upper bounds

% stacked (terminal) state (output) constraints
JbX_e = kron(eye(N),Jbx);
Lbx_e = repmat(lbx,N,1);
Ubx_e = repmat(ubx,N,1);

% populate the output structure
model.sym_x = Y;
model.sym_u = U;
model.sym_p = sym_p;

model.Ad = Ad;
model.Bd = Bd;
model.A = A;
model.B = B;
model.C = Cd;
model.dyn_expr_phi = dyn_expr_phi;

model.cost_expr_ext_cost_0 = cost_expr_ext_cost_0;
model.cost_expr_ext_cost = cost_expr_ext_cost_0;  % no intermediate stage cost
model.cost_expr_ext_cost_e = cost_expr_ext_cost_e;

model.Vu_0 = Vu_0;
model.Vx_0 = Vx_0;
model.W_0 = W_0;
model.y_ref_0 = y_ref_0;

model.Vu = zeros(size(Vu_0));
model.Vx = zeros(size(Vx_0));
model.W = zeros(size(W_0));
model.y_ref = zeros(size(y_ref_0));

model.Vx_e = Vx_e;
model.W_e = W_e;
model.y_ref_e = y_ref_e;

% no need for path constraints
model.Jbu = JbU;
model.lbu = Lbu;
model.ubu = Ubu;
model.Jbx_e = JbX_e;
model.lbx_e = Lbx_e;
model.ubx_e = Ubx_e;
end
