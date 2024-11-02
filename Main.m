%-------------------------------------------------------------------%
%  Main 主函数                                                      %
%-------------------------------------------------------------------%

clear, clc, close all;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum = 5;       % 独立运行的次数
% 算法列表
Algs = ["nca_mpode"];    % 单独测试某个算法
% Algs = ["pso","de","mspso","copso","tlpso","baso","ecsa","issa","pltvaciwpso"];
AlgsNum = length(Algs);
% NCA中权重的k值
% K = [0.3, 0.4, 0.5, 0.6, 0.7];
K = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
% Load dataset
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia"];
DataPath='D:\gj\MATLAB\feature selection\DE-filter\DataSet\';

% Perform feature selection   使用进化算法进行特征选择
plotdata = zeros(AlgsNum,runNum,opts.T);    % 记录每个算法每次运行时每一代的最优解 (三维矩阵)
result = zeros(AlgsNum,opts.T);             % 记录每个算法多次运行后每一代的最优解的均值
SFNum = zeros(AlgsNum,runNum);              % 每个算法每次运行选择的特征数量
Time = zeros(AlgsNum,runNum);               % 每个算法每次运行的时间
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);
KACC = zeros(length(Problem),length(K));

 for k = 1:1%length(K)
    fprintf('\n========== k=%f ==========\n',K(k));
for p = 13 : 18%length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf(">>>>>>>>>>load data: <%s>\n",p_name);
    if p == 8 || p == 13 || p == 16 || p == 18
        ho = 0.2;
    else
        ho = 0.3;
    end
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum
            % Divide data into training and validation sets
            HO = cvpartition(label,'HoldOut',ho); 
            opts.Model = HO; 
            NCA = jNeighborhoodComponentAnalysis(feat,label,K(k));
            ncaFeat = NCA.ff;
            FS = jfs(Algs(i),ncaFeat,label,opts);

            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % 记录每个算法每代的最优解用于收敛图 FS.c为最优解
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(ncaFeat(:,sf_idx),label,opts); % Accuracy   
        end
%         KACC(p,k) = mean(Accuracy(i,:));
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % 多次运行选择特征的平均个数
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % 平均时间
        
    end
% 数据保存

if  AlgsNum == 1 && Algs(1) == "nca_mpode"
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\NCA_MPODE\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\NCA_MPODE\';
else
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\comAlgs\';
end
filename1 = strcat(savepath,p_name,'.mat');
save(filename1,'Accuracy','SFNum','Time','Gbest');
filename2 = strcat(plotpath,p_name,'.mat');
save(filename2,'plotdata');
fprintf("\n %s的数据保存成功！\n",p_name);

end

end
% Kpath = 'E:\MATLAB\feature selection\DE-filter\SaveData\NCA_MPODE\K\';
% filename3 = strcat(Kpath,'diffk','.mat');
% save(filename3,'KACC');
% fprintf("\n K值数据保存成功！\n");


%% 利用互信息和MPDE

clear, clc, close all;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum = 20;       % 独立运行的次数
% 算法列表
Algs = ["nca_mpode"];    % 单独测试某个算法
% Algs = ["pso","de","mspso","copso","tlpso","baso","ecsa","issa","pltvaciwpso"];
AlgsNum = length(Algs);

% Load dataset
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
DataPath='D:\gj\MATLAB\feature selection\DE-filter\DataSet\';

K = [0.5, 0.6, 0.7, 0.8, 0.9];

% Perform feature selection   使用进化算法进行特征选择
plotdata = zeros(AlgsNum,runNum,opts.T);    % 记录每个算法每次运行时每一代的最优解 (三维矩阵)
result = zeros(AlgsNum,opts.T);             % 记录每个算法多次运行后每一代的最优解的均值
SFNum = zeros(AlgsNum,runNum);              % 每个算法每次运行选择的特征数量
Time = zeros(AlgsNum,runNum);               % 每个算法每次运行的时间
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);
KACC = zeros(length(Problem),length(K));

for k = 1 :length(K)
    fprintf('\n========== MI_K=%f ==========\n',K(k));
for p = 1 : length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf(">>>>>>>>>>load data: <%s>\n",p_name);
    if p == 8 || p == 13 || p == 16 || p >= 18
        ho = 0.2;
    else
        ho = 0.3;
    end
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum
            % Divide data into training and validation sets
            HO = cvpartition(label,'HoldOut',ho); 
            opts.Model = HO; 
            MIFeat = MI_TOOL(feat,label,K(k));
            FS = jfs(Algs(i),MIFeat,label,opts);
             
            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % 记录每个算法每代的最优解用于收敛图 FS.c为最优解
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(MIFeat(:,sf_idx),label,opts); % Accuracy   
        end
        KACC(p,k) = mean(Accuracy(i,:));
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % 多次运行选择特征的平均个数
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % 平均时间
        
    end
% 数据保存

if  AlgsNum == 1 && Algs(1) == "nca_mpode"
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\NCA_MPODE\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\NCA_MPODE\';
else
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\comAlgs\';
end
filename1 = strcat(savepath,p_name,'.mat');
save(filename1,'Accuracy','SFNum','Time','Gbest');
filename2 = strcat(plotpath,p_name,'.mat');
save(filename2,'plotdata');
fprintf("\n %s的数据保存成功！\n",p_name);

end

end
Kpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\NCA_MPODE\K\';
filename3 = strcat(Kpath,'MI_K','.mat');
save(filename3,'KACC');
fprintf("\n K值数据保存成功！\n");

%% 对比算法+MPDE
clear, clc, close all;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum = 30;       % 独立运行的次数
% 算法列表
Algs = ["mpde"];    % 单独测试某个算法
% Algs = ["pso","de","mspso","copso","tlpso","baso","ecsa","issa","pltvaciwpso"];
AlgsNum = length(Algs);

% Load dataset
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
DataPath='D:\gj\MATLAB\feature selection\DE-filter\DataSet\';

% Perform feature selection   使用进化算法进行特征选择
plotdata = zeros(AlgsNum,runNum,opts.T);    % 记录每个算法每次运行时每一代的最优解 (三维矩阵)
result = zeros(AlgsNum,opts.T);             % 记录每个算法多次运行后每一代的最优解的均值
SFNum = zeros(AlgsNum,runNum);              % 每个算法每次运行选择的特征数量
Time = zeros(AlgsNum,runNum);               % 每个算法每次运行的时间
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);

for p = 1 : length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf("\n------load data: <%s>------\n",p_name);
    if p == 8 || p == 13 || p == 16 || p >= 18
        ho = 0.2;
    else
        ho = 0.3;
    end
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum
            % Divide data into training and validation sets
            HO = cvpartition(label,'HoldOut',ho); 
            opts.Model = HO; 
            FS = jfs(Algs(i),feat,label,opts);

            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % 记录每个算法每代的最优解用于收敛图 FS.c为最优解
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(feat(:,sf_idx),label,opts); % Accuracy   
        end
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % 多次运行选择特征的平均个数
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % 平均时间
        
    end
% 数据保存

if  AlgsNum == 1 && Algs(1) == "mpde"
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\MPDE\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\MPDE\';
else
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\comAlgs\';
end
filename1 = strcat(savepath,p_name,'.mat');
save(filename1,'Accuracy','SFNum','Time','Gbest');
filename2 = strcat(plotpath,p_name,'.mat');
save(filename2,'plotdata');
fprintf("\n %s的数据保存成功！\n",p_name);

end

%% MPODE+MI
clear, clc, close all;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum = 30;       % 独立运行的次数
% 算法列表
Algs = ["mi_mpode"];    % 单独测试某个算法
% Algs = ["pso","de","mspso","copso","tlpso","baso","ecsa","issa","pltvaciwpso"];
AlgsNum = length(Algs);

% Load dataset
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
DataPath='D:\gj\MATLAB\feature selection\DE-filter\DataSet\';


% Perform feature selection   使用进化算法进行特征选择
plotdata = zeros(AlgsNum,runNum,opts.T);    % 记录每个算法每次运行时每一代的最优解 (三维矩阵)
result = zeros(AlgsNum,opts.T);             % 记录每个算法多次运行后每一代的最优解的均值
SFNum = zeros(AlgsNum,runNum);              % 每个算法每次运行选择的特征数量
Time = zeros(AlgsNum,runNum);               % 每个算法每次运行的时间
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);

for p = 19 : 19 %length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf(">>>>>>>>>>load data: <%s>\n",p_name);
    if p == 8 || p == 13 || p == 16 || p >= 18
        ho = 0.2;
    else
        ho = 0.3;
    end
    if p >=12
        K = 0.5;
    else
        K = 0.8;
    end
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum
            % Divide data into training and validation sets
            HO = cvpartition(label,'HoldOut',ho); 
            opts.Model = HO; 
            MIFeat = MI_TOOL(feat,label,K);
            FS = jfs(Algs(i),MIFeat,label,opts);
             
            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % 记录每个算法每代的最优解用于收敛图 FS.c为最优解
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(MIFeat(:,sf_idx),label,opts); % Accuracy   
        end
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % 多次运行选择特征的平均个数
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % 平均时间
        
    end
% 数据保存

if  AlgsNum == 1 && Algs(1) == "mi_mpode"
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\MI_MPODE\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\MI_MPODE\';
else
    savepath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
    plotpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\PlotData\comAlgs\';
end
filename1 = strcat(savepath,p_name,'.mat');
save(filename1,'Accuracy','SFNum','Time','Gbest');
filename2 = strcat(plotpath,p_name,'.mat');
save(filename2,'plotdata');
fprintf("\n %s的数据保存成功！\n",p_name);

end

%% 消融实验
clear, clc, close all;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
runNum = 30;       % 独立运行的次数
% 算法列表
Algs = ["mpode-v1","mpode-v2"];  %多种群机制，多种群+OBL
AlgsNum = length(Algs);

% Load dataset
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
DataPath='E:\MATLAB\feature selection\DE-filter\DataSet\';

% Perform feature selection   使用进化算法进行特征选择
plotdata = zeros(AlgsNum,runNum,opts.T);    % 记录每个算法每次运行时每一代的最优解 (三维矩阵)
result = zeros(AlgsNum,opts.T);             % 记录每个算法多次运行后每一代的最优解的均值
SFNum = zeros(AlgsNum,runNum);              % 每个算法每次运行选择的特征数量
Time = zeros(AlgsNum,runNum);               % 每个算法每次运行的时间
Accuracy = zeros(AlgsNum,runNum);
Gbest = zeros(AlgsNum,runNum);

for p = 1 : 30 %length(Problem)
    p_name = Problem(p);
    dataname=strcat(DataPath,p_name,'.mat');
    load(dataname); 
    fprintf(">>>>>>>>>>load data: <%s>\n",p_name);
    if p == 8 || p == 13 || p == 16 || p >= 18
        ho = 0.2;
    else
        ho = 0.3;
    end
    
    for i = 1 : AlgsNum
        fprintf("============== %s ==============\n",Algs(i));
        for n = 1 : runNum
            % Divide data into training and validation sets
            HO = cvpartition(label,'HoldOut',ho); 
            opts.Model = HO; 
            FS = jfs(Algs(i),feat,label,opts);
             
            sf_idx = FS.sf;             % Define index of selected features
            plotdata(i,n,:) = FS.c;       % 记录每个算法每代的最优解用于收敛图 FS.c为最优解
            Time(i,n) = FS.t;
            SFNum(i,n) = FS.nf;
            Gbest(i,n) = FS.gb;
            Accuracy(i,n)  = jknn(feat(:,sf_idx),label,opts); % Accuracy   
        end
        fprintf('\n %s: \n Mean of Accuracy: %f\n Std of Accuracy: %f\n\n',...
                Algs(i), mean(Accuracy(i,:)), std(Accuracy(i,:)) );
        fprintf(' Mean of Gbest: %f\n Std of Gbest: %f\n\n',...
                   mean(Gbest(i,:)), std(Gbest(i,:)) );
        fprintf(" Number of Selected Feature: %f \n",mean(SFNum(i,:)));    % 多次运行选择特征的平均个数
        fprintf(" Mean of Times: %f\n",mean(Time(i,:)) );                  % 平均时间
        
    end
% 数据保存


savepath = 'E:\MATLAB\feature selection\DE-filter\SaveData\Ablation\';
plotpath = 'E:\MATLAB\feature selection\DE-filter\SaveData\PlotData\Ablation\';

filename1 = strcat(savepath,p_name,'.mat');
save(filename1,'Accuracy','SFNum','Time','Gbest');
filename2 = strcat(plotpath,p_name,'.mat');
save(filename2,'plotdata');
fprintf("\n %s的数据保存成功！\n",p_name);

end