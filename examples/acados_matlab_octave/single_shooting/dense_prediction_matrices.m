function [F,PHI] = dense_prediction_matrices(A, B, C, Np)

    % Calculation of prediction matrices for dense dynamics
    % Y = F * X0 + PHI * U
    Nc = Np;  % same control and prediction horizon
    
    % Determine number of inputs, outputs and states
    [n_states, n_inputs] = size(B);
    n_outputs = size(C,1);
    
    % System matrices are constant during the horizon, create blocks
    A = repmat(A,[1,1,Np]);
    B = repmat(B,[1,1,Np]);
    
    %% %%%%%% F matrix %%%%%%% %%
    % Determine size of matrix F
    % n_f, m_f -  number of rows and columns of matrix F
    n_f = n_outputs * Np;
    m_f = n_states;

    % Initalize matrix F
    F = nan(n_f, m_f);

    % Fill matrix F one prediction step at the time
    % Use previous results in variable F_aux
    F_aux = A(:,:,1);
    
    F(1:n_outputs,:) = C*F_aux;
    for i = 1 : (Np - 1)
        %Define first and last row
        f_row = (i * n_outputs) + 1;
        l_row = f_row + n_outputs - 1;

        %Calculate matrix based on previous ca
        F_aux = A(:,:,i+1)*F_aux;
        temp2 = C*F_aux;
        for j = f_row : n_outputs : l_row
            for k = 1 : n_outputs
                F(j+k-1, :) = temp2(k,:);
            end
        end
    end


    %% %%%%%% PHI matrix %%%%%%% %%
    % INIT %
    % Determine size of matrix PHI
    % n_p, m_p -  number of rows and columns of matrix Phi
    n_phi = n_outputs * Np;
    m_phi = n_inputs  * Nc;

    % Initalize matrix Phi
    PHI = zeros(n_phi, m_phi);

    % CALCULATE 1 %
    % Caculate first n_inputs columns from matrix F
    % CB, CAB, CA^2B ... 

    for i = 1 : Np
        f_PHI_row = (i - 1) * n_outputs + 1;    % First row of matrix PHI
        l_PHI_row = i * n_outputs;  % Last row of matrix PHI
        f_PHI_col = (min(i,Nc) - 1) * n_inputs + 1; % First column of matrix PHI
        l_PHI_col = min(i,Nc) * n_inputs; % Last column of matrix PHI

        temp3 = C * B(:,:,i);
        for k = f_PHI_row : n_outputs : l_PHI_row
            for x = 1 : n_outputs
                PHI(k+x-1, f_PHI_col : l_PHI_col) = PHI(k+x-1, f_PHI_col : l_PHI_col) + temp3(x,:);    
            end                
        end

        for j = 1 : Np - i
            % Define rows for iteration step of calculation of PHI 
            f_PHI_row = l_PHI_row + 1;    % First row of matrix PHI
            l_PHI_row = f_PHI_row + n_outputs - 1;  % Last row of matrix PHI

            % Calculate predefined rows
            M = MultiplyStateMatrices(A,i+1,i+j);
            temp4 = C* M * B(:,:,i);
            for k = f_PHI_row : n_outputs : l_PHI_row
                for x = 1: n_outputs
                    PHI(k+x-1, f_PHI_col : l_PHI_col) =  PHI(k+x-1, f_PHI_col : l_PHI_col) + temp4(x,:);
                end
            end
        end
    end    
end

function M = MultiplyStateMatrices(A, i, n)
%MULTIPLY STATE MATRICES 
% Functions calculates matrix product M = A(n) * A(n-1) * ... A(i-1) * A(i)  

    M = eye(size(A(:,:,1)));
    for j=i:n
        M = A(:,:,j)*M;
    end

end
