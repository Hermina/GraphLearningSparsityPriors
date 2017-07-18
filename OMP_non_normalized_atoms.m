function [X]=OMP_non_normalized_atoms(D,Y,T0)
%==========================================================================
%    Sparse signals decomposition using OMP
% =========================================================================

% Description: Compute the sparse coding coefficients for a set of signals Y,
% given a dictionary D and a specified sparisty level T0.
% The function is a slight modification of the OMP.m function included in
% the KSVD toolbox that implements the following paper: 
% "The K-SVD: An Algorithm for Designing of Overcomplete Dictionaries for Sparse Representation", 
% written by M. Aharon, M. Elad, and A.M. Bruckstein,  IEEE Trans. On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006.
% Before applying OMP, we normalize the atoms of the dictionary, such that
% they have a unit norm
 
%% Input arguments: 
%       D: the dictionary 
%       Y: the signals to represent
%       T0: the max. number of coefficients for each signal.
%% Output arguments: 
%       X: sparse coefficient matrix.


%==========================================================================

[~,M]=size(Y);
[n,K]=size(D);
X = zeros(K,M);

%--------------------------------
% Normalize the dictionary atoms 
%--------------------------------
norm_Of_Atoms = sqrt(sum(D.^2,1));

zeronorm=find(norm_Of_Atoms==0);
norm_Of_Atoms(zeronorm)=1;
normalized_D = D ./ repmat(norm_Of_Atoms,n,1);

%--------------------------------------------
% Compute the sparse representation using OMP
%--------------------------------------------

for k = 1 : M,
    a = [];
    x = Y(:,k);
    residual = x;
    indx = zeros(T0,1);
    for j=1 : 1 : T0,
       
        proj = normalized_D' * residual;
        [~,pos] = max(abs(proj));
        pos = pos(1);
        indx(j) = pos;
        a = pinv(normalized_D(:,indx(1:j))) * x;
        residual = x - normalized_D(:,indx(1:j)) * a;
        if sum(residual.^2) < 1e-6
            break;
        end
    end;
    temp = zeros(K,1);
    temp(indx(1:j)) = a;
    X(:,k)=sparse(temp);
end;

%--------------------------------
% Renormalize the coefficients
%--------------------------------

X = X ./ repmat(norm_Of_Atoms',1,M);
