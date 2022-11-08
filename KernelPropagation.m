function K = KernelPropagation(L,pairwiseConstraint,PriorConstraint)
% pairwiseConstraint - pairwiseConstraint(i,1:2) �ڽ����
% pairwiseConstraint��i,3������Լ��

n = size(L,1);

%% ����Լ���㵽ǰ��
% �Խ���Ԫ�غͷǶԽ���Ԫ�ص��С��зֱ������˳������ 

IndexConstraintPoint = unique([pairwiseConstraint(:,1) ;pairwiseConstraint(:,2)])';
numConstraintPoint = length(IndexConstraintPoint);%���l

% ��������������
IndexSort = [IndexConstraintPoint  setdiff(1:n ,IndexConstraintPoint)]; % ��������
%C = setdiff(A,B) ���� A �д��ڵ� B �в����ڵ����ݣ��������ظ���,C �������.
L = L(IndexSort,:);
L = L(:,IndexSort)+eps*eye(n); 

%% ����Learned SeedMatrix
%min_X tr(CX) s.t. X>=0,AX(:) = b;
% 
% C =  L(1:numConstraintPoint,1:numConstraintPoint) - L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end)*L(1:numConstraintPoint,numConstraintPoint+1:end)' ;
% 
% numConstraint = size(pairwiseConstraint,1);%Լ���ĸ���
% A = sparse(numConstraintPoint*numConstraintPoint, numConstraint);
% b = sparse( numConstraint,1);
% for i = 1:numConstraint
%     a = sparse(pairwiseConstraint(i,1),pairwiseConstraint(i,2),1,numConstraintPoint,numConstraintPoint);
%     A(i,:) = a(:);
%     b(i) = pairwiseConstraint(i,3);
% end
% addpath(genpath('./YALMIP-master'))
% addpath(genpath('./SeDuMi_1_3'))
% X = sdpvar(numConstraintPoint,numConstraintPoint,'symmetric'); % ���߱���
% Constraints = [X>=0, A*X(:)==b]; % Լ��
% OF = trace(C*X); % Ŀ�꺯��
% options = sdpsettings('verbose',1,'solver','sedumi'); % ���������
% solvesdp(Constraints,OF,options)
% 
% LearnedSeedKernel = sparse(double(X));
LearnedSeedKernel = PriorConstraint;
%% ����full kernel

Klu=-LearnedSeedKernel*L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end);
Kuu=L(numConstraintPoint+1:end,numConstraintPoint+1:end)\L(1:numConstraintPoint,numConstraintPoint+1:end)'*LearnedSeedKernel*L(1:numConstraintPoint,numConstraintPoint+1:end)/L(numConstraintPoint+1:end,numConstraintPoint+1:end);
K = [LearnedSeedKernel,Klu;Klu',Kuu];
% imshow(full(K),[])

%% ��ԭ��Ĵ���

K(IndexSort,:) = K;
K(:,IndexSort) = K;
%% �淶��

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



