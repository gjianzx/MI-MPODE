%% 箱型图
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
    
path1 = 'E:\MATLAB\feature selection\DE-filter\SaveData\MI_MPODE\';
path2 = 'E:\MATLAB\feature selection\DE-filter\SaveData\Algorithms\';
path3 = 'E:\MATLAB\feature selection\DE-filter\SaveData\MPDE\';

opts.T  = 100; 
result = zeros(11,30); 
fprintf("箱型图对比\n");

for i = 21:21  
    p_name = Problem(i);
    % MI_MPODE
    filename1 = strcat(path1,p_name,'.mat');
    load(filename1);
    result(11,:)=Gbest(1,:);

    % 别人的对比算法
    filename2 = strcat(path2,p_name,'.mat');
    load(filename2);    
    result(1:9,:)=Gbest(1:9,:);

    % MPDE
    filename3 = strcat(path3,p_name,'.mat');
    load(filename3);
    result(10,:)=Gbest(1,:);

    figure(i);
    boxplot([log(result(1,:)'), log(result(2,:)')...
             log(result(3,:)'), log(result(4,:)')...
             log(result(5,:)'), log(result(6,:)')...
             log(result(7,:)'), log(result(8,:)')...
             log(result(9,:)'), log(result(11,:)')], 'Labels',...
             {'PSO','DE','MSPSO','COPSO','TLPSO','BASO','ECSA','ISSA','PLTVACIW-PSO','MI-MPODE'});
%      xlabel('datasets');
%      ylabel('Fitness Value');
     xlabel('算法');
     ylabel('适应度值');
     xtickangle(45);
end

%% 收敛图
Problem=["Glass","Wine","Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
    
path1 = 'E:\MATLAB\feature selection\DE-filter\SaveData\PlotData\MI_MPODE\';
path2 = 'E:\MATLAB\feature selection\DE-filter\SaveData\PlotData\comAlgs\';
path3 = 'E:\MATLAB\feature selection\DE-filter\SaveData\PlotData\MPDE\';

opts.T  = 100; 
result = zeros(11,opts.T); 

for i = 22:22%1:length(Problem)
    p_name = Problem(i);
    % MI_MPODE
    filename1 = strcat(path1,p_name,'.mat');
    load(filename1);
    for t = 1 : opts.T
            result(11,t)=mean(plotdata(1,:,t));
    end

    % 别人的对比算法
    filename2 = strcat(path2,p_name,'.mat');
    load(filename2);    
    for j = 1 : 9
        for t = 1 : opts.T
            result(j,t)=mean(plotdata(j,:,t));
        end
    end

    % MPDE
    filename3 = strcat(path3,p_name,'.mat');
    load(filename3);
    for t = 1 : opts.T
         result(10,t)=mean(plotdata(1,:,t));
    end



    figure(i);
    xx = 1:opts.T;
    plot(xx,result(1,xx),'r-*', xx,result(2,xx),'k-o', xx,result(3,xx),'g-^',...
         xx,result(4,xx),'b-d', xx,result(5,xx),'m-s', xx,result(6,xx),'y-v',...
         xx,result(7,xx),'c-s', xx,result(8,xx),'r-d', xx,result(9,xx),'k->',...
         xx,result(11,xx),'m-p','MarkerIndices',1:(opts.T/20):opts.T,'LineWidth',1); %grid on;
    legend("PSO","DE","MSPSO","COPSO","TLPSO","BASO","ECSA","ISSA","PLTVACIW-PSO","MI-MPODE",...
          'Location', 'northoutside','NumColumns', 5);
    xlabel('Number of Iterations');
    ylabel('Fitness Value');
%       xlabel('迭代次数');
%       ylabel('平均适应度值');
end

%% K值
Kpath = 'D:\gj\MATLAB\feature selection\DE-filter\SaveData\MI_MPODE\K\';
K = [0.5, 0.6, 0.7, 0.8, 0.9];
Problem=["Heart","Zoo","Parkinsons","Dermatology","Ionosphere","Lung-cancer",...
         "Movement_libras","Musk1","Arrhythmia","LSVT","SCADI","Madelon","Yale","Colon","TOX_171","Leukemia",...
         "ALLAML","GLI_85","Prostate_GE","arcene"];
     
filename = strcat(Kpath,'MI_K','.mat');
load(filename);

xx = 3:22;
y1 = KACC(:,1); y2 = KACC(:,2); y3 = KACC(:,3); y4 = KACC(:,4); y5 = KACC(:,5);
plot(xx,y1,'r-*', xx,y2,'k-o', xx,y3,'g-^', xx,y4,'b-d', xx,y5,'m-s','LineWidth',1);grid on;
xlabel('Datasets');
ylabel('Accurary');
legend("k=0.5","k=0.6","k=0.7","k=0.8","k=0.9");
