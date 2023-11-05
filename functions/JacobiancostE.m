function [F,J]=JacobiancostE(x, param)
    nimg = length(param.uv); % Number of camera poses.
    uv = param.uv;
    K = param.K;  
    
    % Extract R, T, X
    [Rvec,Tvec,X] = deserialize(x,nimg);
    nX = length(X); nXn=0;
    for i=1:nimg
        nXn = nXn + length(uv{i}); end
    
    % dx = inv(J'*J) * (J'*[b-f(x)])
    % 1. F = b-f(x)
    F = zeros(2*nXn,1); 
    
    count = 1;
    for i = 1:nimg        
        % Rotation, Translation, [X, Y, Z]
        X_idx = uv{i}(4,:); nXi = size(X_idx, 2);
        R = RotationVector_to_RotationMatrix(Rvec(:,i)); T = Tvec(:,i); Xi = X(:,X_idx);   
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % write code to calculate reprojection errors and store them into
        % variable F
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        end
    end
        
    % 2. J
    % Jacobian Matrix
    % Calculation derivate on Jacobian_derivative.m
    Jcell = cell(nimg,1); Jrow = 6*nimg+3*nX;
    J = [];
        
    for i=1:nimg
        X_idx = uv{i}(4,:); nXi = size(X_idx, 2);
        Jcell{i} = sparse( zeros(2*nXi, Jrow) );
        
        % Rotation, Translation, [X, Y, Z, 1]
        R = Rvec(:,i); T = Tvec(:,i); Xi = X(:,X_idx);  
        parameters = [K(1,1); K(2,2); K(1,2); K(1,3); K(2,3); R(1); R(2); R(3); T(1); T(2); T(3)];
        count = 1;
        for j=1:nXi            
            pFpP = JacobianPose2(Xi(:,j), parameters);
            pFpX = JacobianPoint(Xi(:,j), parameters);
            Jcell{i}(count : (count+1), :) = sparse( [zeros(2, 6 * (i - 1))        pFpP zeros(2, 6 * (nimg-i))...
                                                      zeros(2, 3 * (X_idx(j) - 1)) pFpX zeros(2, 3 * (nX - X_idx(j)))] );
            
            count = count + 2;           
        end
    end
    
    for i=1:nimg
        J = [J; Jcell{i}];
    end

    %Jshow = J(1:300,1:300);
    %Jshow2 = Jcell{param.key1}(1:300,1:300);
end