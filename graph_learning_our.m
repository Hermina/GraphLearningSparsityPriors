function [learned_Laplacian, x] = graph_learning_our(alpha, beta, maxEpoch, param)
%learns a graph from signals with a sparse representation. Parameters:
%alpha: gradient descent step
%beta: parameter imposing weight matrix sparsity
%maxEpoch: maximum epoch of gradient descent
%param: must have...
%   -N: number of nodes
%   -K: max power of the polynomial dictionary
%   -S: number of subdictionaries
%   -y: signals
%   -T0: sparsity of the signal representation
%   -alpha: polynomial coefficients

verbose=0;

%initialise a random weight matrix
[learned_Laplacian, learned_W] = init_by_weight(param.N);

eye_N = eye(param.N);
for epoch =1:maxEpoch
    %construct the dictionary from the newly learned graph
    if(verbose)
        disp(['Epochs remaining: ' num2str(maxEpoch-epoch)]);
    end
    for k=0 : max(param.K)
        learned_Laplacian_powers{k + 1} = learned_Laplacian^k;
    end

    for i=1:param.S
        learned_dict{i} = zeros(param.N);
    end

    for k = 1 : max(param.K)+1
        for i=1:param.S
            learned_dict{i} = learned_dict{i} + param.alpha{i}(k)*learned_Laplacian_powers{k};
        end
    end

    learned_dictionary = [learned_dict{1}, learned_dict{2}];

    %------------------------------------------------------------------ 
    %%------------------------ Learning ------------------------------
    %------------------------------------------------------------------ 
    
    %optimizing with regard to x
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);

    %compute signal estimation
    estimated_y=learned_dictionary*x; 
    
    %computing the gradient
    K=max(param.K);
    der_all_new = zeros(param.N, param.N);
    learned_D = diag(sum(learned_W,2));
    learned_D_powers{1} = learned_D^(-0.5);
    learned_D_powers{2} = learned_D^(-1);
    for s=1:param.S
        for k=0:K
            C=zeros(param.N,param.N);
            B=zeros(param.N,param.N);
            for r=0:k-1 
                A = learned_D_powers{1}*learned_Laplacian_powers{k-r}*x((s-1)*param.N+1:s*param.N,:)*(estimated_y - param.y)'*learned_Laplacian_powers{r+1} * learned_D_powers{1};
                B=B+learned_D_powers{1}*learned_W*A*learned_D_powers{1};
                C=C-2*A';
                B=B+A*learned_W*learned_D_powers{2};
            end
            B = ones(size(B)) * (B .* eye_N);
            C = param.alpha{s}(k+1)*(C+B);
            der_all_new = der_all_new + C;
        end            
    end
    %adding the sparsity term gradient
    der_all_new = der_all_new +  beta*sign(learned_W); 
    %making derivative symmetric and removing the diag (that we don't want to change)
    der_sym = (der_all_new + der_all_new')/2 - diag(diag(der_all_new)); 
    
    %gradient descent, adjusting the weights with each step
    alpha = alpha * (0.1^(1/maxEpoch));
    %beta = beta * (10^(1/maxEpoch));
    learned_W = learned_W - alpha * der_sym;
    
    %producing a valid weight matrix
    learned_W(learned_W<0)=0;
    
    % combinatorial Laplacian
    learned_L = diag(sum(learned_W,2)) - learned_W;
    % normalized Laplacian
    learned_Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);
    err(epoch) = app_error(param.y, learned_dictionary, x);
end
figure()
plot(err)
end