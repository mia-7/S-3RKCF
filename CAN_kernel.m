% min_{A>=0, A*1=1, F'*F=I}  trace(D'*A) + r*||A||^2 + 2*lambda*trace(F'*L*F)
% written by Feiping Nie on 2/9/2014
function [A,r] = CAN_kernel(S_hat,X, U, V, K, alpha, beta, k, r, islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% r: paremeter, which could be set to a large enough value. If r<0, then it is determined by algorithm with k
% islocal: 
%           1: only update the similarities of the k neighbor pairs, faster
%           0: update all the similarities
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations

% For more details, please see:
% Feiping Nie, Xiaoqian Wang, Heng Huang. 
% Clustering and Projected Clustering with Adaptive Neighbors.  
% The 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), New York, USA, 2014.

NITER = 30;
num = size(X,1);
if nargin < 10
    islocal = 0;
end
if nargin <9
    r = -1;
end

UV = U*V';
distX = L2_distance_1(UV,UV);%(V',V')
[distX1, ~] = sort(distX,2);
[~, idx] = sort(-2*alpha*K/beta+distX,2);%-K
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k))); %r取值
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);%类似于初始化
end

if r <= 0
    r = mean(rr);
end
lambda = 4*alpha/beta;
for iter = 1:NITER
    A = zeros(num);
    for i=1:num %样本个数
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:num;
        end
        p=0;
        for j=idxa0
            p =p+1;
            dki(p) = -K(i,j);%K(i,i)- 2*K(i,j) + K(j,j)
        end 
        dxi = distX(i,idxa0);
        ad = S_hat(i,idxa0) -(dxi+lambda*dki)/(2*r+eps);% 
        A(i,idxa0) = EProjSimplex_new(ad); %最小化目标函数
    end

    A = (A+A')/2; %自适应近邻得到的不是对称矩
end




