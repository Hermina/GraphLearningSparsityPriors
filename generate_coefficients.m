function [ original_alpha ] = generate_coefficients( param )
%Generates coefficients for the kernels of polynomial dictionary as a
%Taylor approximation to heat diffusion.

for j=1:param.S
    for i=0:max(param.K)
        original_alpha{j}(i + 1) = ((-param.t(j))^(i))/factorial(i);
    end
end

%this will invert the graph of the polynomial -> we want to be able to
%efficiently represent all frequencies of the signal (we're approximating
%1-heat_kernel here)
original_alpha{2}=-original_alpha{2};
original_alpha{2}(1) = original_alpha{2}(1) + 1;
end

