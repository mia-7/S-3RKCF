function [K] = KerP(L,IndexLabeledSmps,label)

Nl=size(IndexLabeledSmps,1);
PriorConstraint = zeros(Nl);
for i = 1:Nl
    for j = 1:Nl
        PriorConstraint(i,j) = (label(i)==label(j))*1;
    end
end

[I J Val] = find(PriorConstraint+1e-5); % 转化为稀疏格式
SparsepriorConstraint = [I J Val-1e-5];
K = KernelPropagation(L,SparsepriorConstraint,PriorConstraint);