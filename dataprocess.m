clc;
close all;
clear;
problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "gisette","ALLAML","GLI_85","Prostate_GE","arcene"];
% 导入源数据文件
OriginDataPath='E:\MATLAB\feature selection\DE-filter\OriginalData\';
DataPath='E:\MATLAB\feature selection\DE-filter\DataSet\';
% num=length(problem);
num = 9;

%% 处理数据
for i = 19:23
    p_name=problem(i);
    fprintf("读取数据：%s\n",p_name);
    %源数据文件路径
%     datapath=strcat(OriginDataPath,p_name,'.txt');
     datapath=strcat(OriginDataPath,p_name,'.mat');  % Yale,TOX-171, Madelon ,Colon,Leukemia

    %载入数据
    data = load(datapath);
    % 保存mat文件路径
    savepath=strcat(DataPath,p_name,'.mat');
    
    if  i == 2 || i==8
        label = data(:,1);   %label在文件的第一列
        feat = data(:,2:end);    
    elseif i==1 || i==3 || i==4 || i==6 || i==9 || i==11 || i==13  
        label = data(:,end); %label在文件的最后一列
        feat = data(:,1:end-1);
    elseif i==12            % 处理excel，LSVT
        datapath=strcat(OriginDataPath,p_name,'.xlsx');  % LSVT
        feat = zeros(126,310);
        for n = 2:127
            S1 = "A" + n;
            S2 = "KX" + n;
            S = S1 + ":" + S2;
            data1(n,:) = xlsread(datapath,'Data',S);
            feat(n-1,:) = data1(n,:);
            
        end
            data2 = xlsread(datapath,'Binary response','A2:A127');
            label = data2;
            
    elseif i==5             % Parkinsons
        feat = zeros(195,22);
        label = data(:,17);
        for i = 1:195
            for j = 1:23               
                if j ~=17
                  feat(i,j) = data(i,j);
                end
                
            end
        end
        feat(:,17)=[];
    else                             % X,Y转换为feat,label
        feat=data.X;
        label=data.Y;
    end
    save(savepath,'label','feat');
    fprintf("数据保存成功！\n");
end
