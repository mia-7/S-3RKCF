function K = KernelPropagation(L,pairwiseConstraint,PriorConstraint)
% pairwiseConstraint - pairwiseConstraint(i,1:2) 邻接序号
% pairwiseConstraint（i,3）先验约束

n = size(L,1);

%% 排序约束点到前面
% 对角线元素和非对角线元素的行、列分别按照相对顺序排列 

IndexConstraintPoint = unique([pairwiseConstraint(:,1) ;pairwiseConstraint(:,2)])';
numConstraintPoint = length(IndexConstraintPoint);%求出l

% 计算点的排序索引
IndexSort = [IndexConstraintPoint  setdiff(1:n ,IndexConstraintPoint)]; % 排序索引
%C = setdiff(A,B) 返回 A 中存在但 B 中不存在的数据，不包含重复项,C 是有序的.
L = L(IndexSort,:);
L = L(:,IndexSort)+eps*eye(n); 

%% 计算Learned SeedMatrix
%min_X tr(CX) s.t. X>=0,AX(:) = b;
% 
% C =  L(1:numConstraintPoint,1:numConstraintPoint) - L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end)*L(1:numConstraintPoint,numConstraintPoint+1:end)' ;
% 
% numConstraint = size(pairwiseConstraint,1);%约束的个数
% A = sparse(numConstraintPoint*numConstraintPoint, numConstraint);
% b = sparse( numConstraint,1);
% for i = 1:numConstraint
%     a = sparse(pairwiseConstraint(i,1),pairwiseConstraint(i,2),1,numConstraintPoint,numConstraintPoint);
%     A(i,:) = a(:);
%     b(i) = pairwiseConstraint(i,3);
% end
% addpath(genpath('./YALMIP-master'))
% addpath(genpath('./SeDuMi_1_3'))
% X = sdpvar(numConstraintPoint,numConstraintPoint,'symmetric'); % 决策变量
% Constraints = [X>=0, A*X(:)==b]; % 约束
% OF = trace(C*X); % 目标函数
% options = sdpsettings('verbose',1,'solver','sedumi'); % 求解器设置
% solvesdp(Constraints,OF,options)
% 
% LearnedSeedKernel = sparse(double(X));
LearnedSeedKernel = PriorConstraint;
%% 计算full kernel

Klu=-LearnedSeedKernel*L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end);
Kuu=L(numConstraintPoint+1:end,numConstraintPoint+1:end)\L(1:numConstraintPoint,numConstraintPoint+1:end)'*LearnedSeedKernel*L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end);
K = [LearnedSeedKernel,Klu;Klu',Kuu];
% imshow(full(K),[])

%% 还原点的次序

K(IndexSort,:) = K;
K(:,IndexSort) = K;
%% 规范化

% % % % % D = zeros(size(K,1),1);
% % % % % for i=1:size(K,1)
% % % % %     D(i) = sum(K(i,:));
% % % % % end
% % % % % NS = zeros(size(K,1),size(K,2));
% % % % % for i=1:size(K,1)
% % % % %     NS(i,:) = K(i,:) / sqrt(D(i));
% % % % % end
% % % % % for j=1:size(K,2)
% % % % %     K(:,j) = NS(:,j) / sqrt(D(j));
% % % % % end



