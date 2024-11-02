%------------------------------------%
%  最终结果保存到excel                %
%------------------------------------%

clc;
clear;
close all;
path = 'E:\MATLAB\feature selection\DE-filter\SaveData\Algs_datas.xlsx';
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
% Algs = ["MI_MPODE"];   
% Algs = ["PSO","DE","MSPSO","COPSO","TLPSO","BASO","ECSA","ISSA","PLTVACIW_PSO"];
Algs=["MPODE-v1","MPODE-v2"];  % 消融实验所需数据

% 导入对比算法的mat数据
if length(Algs) == 1 && Algs(1) == "MI_MPODE"
    Algs_path = 'E:\MATLAB\feature selection\DE-filter\SaveData\MI_MPODE\';
    volume=["L"];
    fprintf("\n Loading MI_MPDE Data \n");
elseif length(Algs) == 1 && Algs(1) == "MPDE"
    Algs_path = 'E:\MATLAB\feature selection\DE-filter\SaveData\MPDE\';
    volume=["B"];
    fprintf("\n Loading MPDE Data \n");
elseif length(Algs) == 2 && Algs(1) == "MPODE-v1"
    Algs_path = 'E:\MATLAB\feature selection\DE-filter\SaveData\Ablation\';
    volume=["M","N"];
    fprintf("\n Loading Ablation Data \n");
else
    Algs_path = 'E:\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
    volume=["C","D","E","F","G","H","I","J","K"];
    fprintf("\n Loading Comparison Algorithm Data \n");
end

% 写入数据集名称
for p = 1 : length(Problem)
    p_name = Problem(p);
    fprintf("\n load %s >>>>\n",p_name);
    filename = strcat(Algs_path,p_name,'.mat');
    load(filename);
    Type = [p_name,"Mean","Std"];
    for i = 1:length(Type)
        S1 = "A" + (i + 3 * (p-1));
        writecell({Type(i)},path,'Sheet',1,'Range',S1);
        writecell({Type(i)},path,'Sheet',4,'Range',S1);
    end
    SS1 = "A" + (p+1); 
    writecell({p_name},path,'Sheet',2,'Range',SS1);
    writecell({p_name},path,'Sheet',3,'Range',SS1);
    
    % 写入算法名称和数据  判断是否自己的算法还是对比算法
    for i = 1 : length(Algs)
        S2 = volume(i) + (1 + (p - 1) * 3);
        SS2 = volume(i) + 1;
        writecell({Algs(i)},path,'Sheet',1,'Range',S2);              % 先写入算法名称
        writecell({Algs(i)},path,'Sheet',2,'Range',SS2);
        writecell({Algs(i)},path,'Sheet',3,'Range',SS2);
        writecell({Algs(i)},path,'Sheet',4,'Range',S2);
        Acc_result = [mean(Accuracy(i,:)), std(Accuracy(i,:))];
        Gb_result = [mean(Gbest(i,:)),std(Gbest(i,:))];
        for j = 1 : length(Acc_result)
            S3 = volume(i) + ( (j+1)+(p-1) * 3 );
            writecell({Acc_result(j)},path,'Sheet',1,'Range',S3);    % 写入精度数据
            writecell({Gb_result(j)},path,'Sheet',4,'Range',S3);     % 写入最优解数据
        end
        F_result = mean(SFNum(i,:));
        T_result = mean(Time(i,:));
        S4 = volume(i) + (p+1);
        writecell({F_result},path,'Sheet',2,'Range',S4);   % 特征数量数据
        writecell({T_result},path,'Sheet',3,'Range',S4);   % 计算时间数据
    end
end
fprintf("\n Successfully save excel data！\n");

