function [learned_Laplacian, W] = init_by_weight(n)
%creates a valid weight matrix with random entries between 0 and 1 and
%returns that with the corresponding normalised Laplacian

W = rand(n,n); %between 0 and 1
for i = 1: n - 1
    for j=(i+1):n
        W(i,j) = W(j,i);
    end
    W(i, i) = 0; 
end
W(n,n) = 0;

L = diag(sum(W,2)) - W; % combinatorial Laplacian
learned_Laplacian = (diag(sum(W,2)))^(-1/2)*L*(diag(sum(W,2)))^(-1/2); % normalized Laplacian

end
