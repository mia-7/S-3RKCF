close all;clear;clc
load('C:/Users/cyj/Desktop/data/YaleB.mat');
data = NormalizeFea(fea',1);
gnd = gnd';
perclass = [];%
nclass = 38;
for i = 1:nclass
    if ~isempty(find(gnd==i, 1))
        perclass(i) = sum(gnd==i);
    end
end
feaSet = data';
labeled_sample = 0.1;
gd_truth = gnd;
nTotal = 2414;


alpha = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5];
beta = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3];

parfor i = 1:length(alpha)
    AC = zeros(1, length(beta));
    MI = zeros(1, length(beta));
    datacht = [];
    datacht.k = nclass;
    datacht.N = nTotal;
    
    KPCFoptions = [];
    KPCFoptions.maxIter = 5;%%%%
    KPCFoptions.NeighborMode = 'KNN';
    KPCFoptions.k = 3;
    KPCFoptions.WeightMode = 'Binary';
    
    options = [];
    options.maxIter = KPCFoptions.maxIter;
    options.k = 3;
    
    U = abs(rand(datacht.N,datacht.k));
    V = abs(rand(datacht.N,datacht.k));
    for j = 1:length(beta)
        acc = [];
        nmi = [];
        label_index = cell(1, nclass);
        for e=1:10
            for ii=1:nclass
                label_index{ii} = randperm(perclass(ii) , ceil(labeled_sample*perclass(ii)));%
            end
            indices_num = length(cell2mat(label_index));
            datacht.indices = reshape([1:indices_num],indices_num,1);%label_index;
            XL =[];
            Xu = [];
            for p = 1:nclass
                if p==1
                    XL = [XL feaSet(:,(label_index{p}))]; %%固定非random选取
                    Xu = [Xu feaSet(:,setdiff(1:perclass(p),label_index{p}))];
                else
                    XL = [XL feaSet(:,(label_index{p}+sum(perclass(1:p-1))))]; %%固定非random选取
                    Xu = [Xu feaSet(:,setdiff((1+sum(perclass(1:p-1))):sum(perclass(1:p)),(label_index{p}+sum(perclass(1:p-1)))))];
                end
            end
            XL =XL';
            Xu =Xu';
            datacht.data = [XL;Xu];
            
            label = [];
            for p = 1:2*nclass
                if p<=nclass
                    if p==1
                       label = [label;gd_truth(label_index{p})];
                    else
                       label = [label;gd_truth(label_index{p}+sum(perclass(1:p-1)))];
                    end
                else
                    if p==(nclass+1)
                       label = [label;gd_truth(setdiff(1:perclass(p-nclass),label_index{p-nclass}))];
                    else
                       label = [label;gd_truth(setdiff((1+sum(perclass(1:p-nclass-1))):sum(perclass(1:p-nclass)),label_index{p-nclass}+sum(perclass(1:p-nclass-1))))];
                    end
                end
            end
            datacht.label = label(1:indices_num,:);
            label(1:indices_num,:)=[];
            datacht.testlabel = label;
            %============init K==========
            UV=U*V';
            IUV=(eye(datacht.N)-UV)'*(eye(datacht.N)-UV);
            %IUV=(eye(datacht.N)-UV)*(eye(datacht.N)-UV)';
            KPoptions = [];
            KPoptions.indices=datacht.indices;%索引
            KPoptions.labelsub=datacht.label;
            
            S = constructW(datacht.data,KPCFoptions);
            S=S/3;
            S=(S+S')/2;
%             S = NormalizeFea(S,1);
%             S = mapminmax(S,0,1);
            
            DCol = sum(S,2);
            D = diag(DCol);
            L = D - S;
            K = KerP(L,datacht.indices,datacht.label);
            datacht.K = K;
            KPoptions.Kinit=datacht.K;
            [ U_final, V_final, obj_out,K_final,S_final] = KPCF_CAN(datacht, K, S, options, KPoptions, U, V,alpha(i),beta(j));
%             [U_final, V_final, obj_out,K_final,S_final] = unsupervised_KPCF(datacht, K, S, options, KPoptions, U, V,alpha(i),beta(j));
%             clustering results
            label = kmeans(V_final,datacht.k,'Replicates',20);
            label(1:indices_num,:)=[];
            label = bestMap(datacht.testlabel,label);
            acc(e) =length(find(datacht.testlabel == label))/length(datacht.testlabel);
           nmi(e)=MutualInfo(datacht.testlabel,label);
%             acc = [acc;ac];
        end
        AC(j) = mean(acc);
        MI(j) = mean(nmi);
    end
    ACC(i,:)= AC;
    NMI(i,:)= MI;
end