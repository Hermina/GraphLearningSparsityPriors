%%
clear all
close all
addpath(genpath(pwd))

%load data in variable X here (it should be a matrix #nodes x #signals)

param.N = size(X,1); % number of nodes in the graph
param.S = 2;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
param.K = [15 15]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);

%% generate dictionary polynomial coefficients from heat kernel
param.t(1) = 2; %heat kernel coefficients
param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
param.alpha = generate_coefficients(param);
disp(param.alpha);


%% initialise learned data
param.T0 = 6; %sparsity level (# of atoms in each signals representation)
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs
for big_epoch = 1:500
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimise with regard to W 
    maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
    beta = 10^(-2); %graph sparsity penalty
    old_L = param.Laplacian;
    [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param,learned_W, learned_W);
    [learned_dictionary, param] = construct_dict(param);
    alpha = alpha*0.985; %gradient descent decreasing
end

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = 4*29;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);
