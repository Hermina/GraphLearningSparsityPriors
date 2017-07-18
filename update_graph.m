function [learned_Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param,original_Laplacian, learned_W) 
%graph updating step by gradient descent

param.Laplacian = param.Laplacian;
eye_N = eye(param.N);
for epoch =1:maxEpoch
     %compute signal estimation
    [learned_dictionary, param] = construct_dict(param);
    estimated_y=learned_dictionary*x; 
    error(epoch) = sum(sum(abs(estimated_y-param.y))) + beta*sum(sum(abs(learned_W)));
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
                A = learned_D_powers{1}*param.Laplacian_powers{k-r}*x((s-1)*param.N+1:s*param.N,:)*(estimated_y - param.y)'*param.Laplacian_powers{r+1} * learned_D_powers{1};
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
    param.Laplacian = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);

end
learned_Laplacian = param.Laplacian;
end

