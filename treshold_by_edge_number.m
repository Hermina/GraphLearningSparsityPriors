function [ tresholded_L ] = treshold_by_edge_number( learned_L, real_edges)
%treshold the graph to have exactly #real_edges edges
tresholded_L = learned_L;
first_crossing=1;
for i = 1 : 300
    Laplacian = learned_L;
    mytol = i/1000;
    Laplacian(Laplacian>-mytol & Laplacian<0.5) = 0;

    estimLowTri = logical(tril(Laplacian,-1)~=0);
    
    % Total number of estimated edges
    num_of_edges = sum(sum(estimLowTri));
    
    if (num_of_edges<=real_edges && first_crossing)
        first_crossing = 0;
        tresholded_L = Laplacian;
        break;
    end
end

end

