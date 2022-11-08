function [ U_final, V_final, obj_out,K_final,S_final] = KPCF_CAN(datacht, K, S, options, KPoptions, U, V,alpha,beta)
% input:
%   datacht: variables about dataset
%       datacht.label: labels of labeled samples
%       datacht.testlabel: labels of unlabeled samples
%       datacht.N: the number of all samples
%       datacht.k: the number of classes
%       datacht.data: the features of all samples  
%   K: predefined kernel matrix (refer to the paper)
%   S: predefined affinity matrix constructed by KNN (refer to the paper)
%   options: 
%       options.k: the number of adaptive neighbor
%       optons.maxIter: the number of outer iteration
%   KPoptions: variables about Kernel propagation
%       KPoptions.labelsub: labels of labeled samples
%       KPoptions.indices: indices of labeled samples
%   U: predefined random matrix of datacht.N*datacht.k
%   V: predefined random matrix of datacht.N*datacht.k
%   alpha: the weight coefficient of measuring the smoothness of kernel 
%   beta: the weight coefficient of adaptive neighbor

% For more details, please see:
%   @article{wu2022self,
%   title={Self-representative kernel concept factorization},
%   author={Wu, Wenhui and Chen, Yujie and Wang, Ran and Ou-Yang, Le},
%   journal={Knowledge-Based Systems},
%   pages={110051},
%   year={2022},
%   publisher={Elsevier}
% }
% If you find the resource useful, hope cite our paper as mentioned above. Thanks. 

obj_out=[];
maxIter = options.maxIter;
labelsub=KPoptions.labelsub;
indices=KPoptions.indices;
k = options.k;

Norm = 2;
NormV = 1;
nSmp = datacht.N;
DCol = sum(S,2);
D = diag(DCol);
L = D-S;
S_hat = S;

[U,V] = NormalizeUV(K, U, V, NormV, Norm);


nIter = 1;
while(nIter<=maxIter)
    
    inerIter=0;
  
    % ===================== update V ========================
    while inerIter<50 
        
        Kp=(abs(K)+K)/2; %K-positive
        Kn=(abs(K)-K)/2; %K-negative
        % ===================== update V ========================
        KU = K*U;               % n^2k
        KUp = Kp*U;
        KUn = Kn*U;
        UKU = U'*KU;            % nk^2
        UKUp = U'*KUp;
        UKUn = U'*KUn;
        VUKUp= V*UKUp;
        VUKUn= V*UKUn;
        DV = D*V;
        WV = S*V;
        UU = U'*U;
        VUKUp = VUKUp + beta*DV*UU;
        VUKUn = VUKUn + beta*WV*UU;
        Vnor=KU+(KU.^2+4*VUKUp.*VUKUn).^0.5;
        V = V.*(Vnor./max(2*VUKUp,1e-10));
        clear  KU KUp KUn UKU UKUp UKUn VUKU VUKUp VUKUn Vnor UU VUU DVUU WVUU;
        
        % ===================== update U ========================
        KV = K*V;               % n^2k
        VV = V'*V;  
        UV = U*V';
        KUVVp = Kp*U*VV;
        KUVVn = Kn*U*VV;
        KUVVp = KUVVp + beta*UV*DV;
        KUVVn = KUVVn + beta*UV*WV;
        Unor = KV+(KV.^2+4*KUVVp.*KUVVn).^0.5;
        U = U.*(Unor./max(2*KUVVp,1e-10));
        clear Kp Kn KV VV KUVV KUVVp KUVVn Unor UVDV UVWV UV UU DV WV;
        
        inerIter=inerIter+1;
    end
%     label = kmeans(V,datacht.k,'Replicates',20);
%     label(1:length(datacht.label),:)=[];
%     label = bestMap(datacht.testlabel,label);
%     acc(nIter) = length(find(datacht.testlabel == label))/length(datacht.testlabel);
    %=================== update W =====================chen
    [S , r] = CAN_kernel(S_hat, datacht.data,U,V,K,alpha,beta,k);
    DCol = sum(S,2);%a = sparse(X)得到X的非零元素及其索引，full(a)得到全矩阵
    D = diag(DCol);%spdiags(A,d,B)将d指定的B中对角线替换成Dcol的列
    L = D - S; 
    lamda = r*beta/2;
    % ===================== update K ======================== WU 
    UV = U*V';
    IUV=(eye(nSmp)-UV)*(eye(nSmp)-UV)';
    A = alpha*L+IUV ;
    K = KerP(A ,indices,labelsub);
    clear  UV IUV
    
    [objNMF, objLap, objKP] = CalculateObj(alpha,beta,lamda,K, U, V, S, L,S_hat);
    obj = objNMF + objLap+ objKP;
    obj_out=[obj_out obj]; 
    nIter = nIter + 1;
    
end
U_final = U;
V_final =V;
K_final = K;
S_final = S;
[U_final,V_final] = NormalizeUV(K, U_final, V_final, NormV, Norm);



%==========================================================================

function [obj_NMF, obj_Lap, obj_KP] = CalculateObj(alpha,beta,lamda,K, U, V, S, L,S_hat)

UK = U'*K; 
UKU = UK*U; 
VUK = V*UK; 
VV = V'*V; 
UV=U*V';
obj_NMF = sum(diag(K))-2*sum(diag(VUK))+sum(diag(K*U*VV*U'));
obj_Lap = beta*sum(diag(UV*L*V*U'))+lamda*sum(diag((S-S_hat)'*(S-S_hat)));
obj_KP = alpha*sum(diag(L*K));


function [U, V] = NormalizeUV(K, U, V, NormV, Norm)
k = size(U,2);
if Norm == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sqrt(sum(U.*(K*U),1)))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
else
    if NormV
        norms = max(1e-15,sum(abs(V),1))';
        V = V*spdiags(norms.^-1,0,k,k);
        U = U*spdiags(norms,0,k,k);
    else
        norms = max(1e-15,sum(U.*(K*U),1))';
        U = U*spdiags(norms.^-1,0,k,k);
        V = V*spdiags(norms,0,k,k);
    end
end

function [K] = NormalizeK(K)
        n = size(K,2);
        norms = max(1e-15,sqrt(diag(sqrt(K.^2))));
        K = spdiags(norms.^-1,0,n,n)*K*spdiags(norms.^-1,0,n,n);

