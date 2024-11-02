clc;
clear;
close all;
%% 重叠柱状图
% 创建示例数据
data1 = [111.30, 190.93, 165.20, 97.63, 359.57, 671.80, 1035.23, 3943.40, ...
         3041.97, 3756.17, 3698.50, 5693.80, 12146.27];
data2 = [72.13, 66.83, 6.47, 17.03, 230.67, 266.67, 63.00, 2090.07,...
         134.73, 119.40, 54.07, 730.60, 1181.03];
data3 = [68.13, 80.27, 9.50, 21.00, 173.93, 290.27, 45.50, 1763.23,... 
         131.77, 238.30, 129.17, 826.10, 1101.17];
data4 = [56.20, 65.60, 6.37, 12.57, 84.53, 160.70, 25.67, 878.23, ...
         69.07, 75.97, 69.67, 455.97, 485.53];
x = 1:numel(data1);
w1=0.8;
% 绘制第一个柱状图
bar(x, log(data1), w1,'FaceColor',"#44045A");
hold on; % 保持图形窗口打开，以便后续绘制
w2=0.6;
% 绘制第二个柱状图，叠加在第一个柱状图之上
bar(x, log(data2),w2, 'FaceColor',"#30688D");
w3=0.4;
% 绘制第三个柱状图，叠加在前两个柱状图之上
bar(x, log(data3),w3, 'FaceColor',"#35B777");
w4=0.2;
bar(x, log(data4),w4, 'FaceColor',"#F8E620");
% 添加图例和标签
legend('DE','MPDE','MPODE','MI-MPODE');
% xlabel('Datasets');
% ylabel('Number of features(Log)');
xlabel('数据集');
ylabel('特征数量(Log)');

ax = gca;
ax.XTickLabels = {'Musk1','Arrhythmia','LSVT','SCADI','Madelon','Yale','Colon','TOX-171','Prostate\_GE',...
                  'Leukemia','ALLAML','arcene','GLI\_85'};
ax.XTickLabelRotation = 45;
 
% 可选的调整图形属性
xlim([0.5, numel(data1)+0.5]); % 设置 x 轴范围
grid on; % 添加网格线

%%
a=zeros(3,3,3);
for i = 1:3
    for j= 1:3
        for k=1:3
            a(i,j,k)=j;
        end
    end
end
bb=mean(a(1,1,:))
%%
a=[3,5,7,9,2];
b=[23,15,8,12,5];

% x=a(randi(length(a),1))
[value,index]=sort(a);
% [~,rank]=sort(index)
c = randperm(6,3)
aa = c(randperm(length(c),1))

%% 指定空间内随机产生一个随机数
a = ones(1,5)*2;
b = ones(1,5)*2;
c = a.*b
FL = rand(5,1).*0.4+0.5
%%
a=zeros(2,1,2);
b=ones(1,2);
c=zeros(2,2);
for i = 1:2
    for j = 1:1
        a(i,j,:) = b(j,:);
    end
end
a(1,1,:)
c(2,:)=a(1,1,:); 
c
%%
b=[23,15,8,12,5];
p_name = 'a';
path = 'E:\MATLAB\feature selection\BDE-FS\';
name=strcat('A','_',p_name,'.mat');
save([path,name],'b');

%% 混沌映射
% Logistic 映射
mu=3.0;        %Growth rate parameter
x=rand(1,100)*(1-0.5)+0.5; %Create a matrix 1*200
% Iterative 200 times
for n=1:100 
    x(n+1)= mu * x(n) * (1- x(n));
end
plot(x(1,:),'k','MarkerSize',10);
xlabel('n');
ylabel('x(n)');

%%
a = randi([1 3])
%% 判断数组中是否包含某个元素
a=[3,4,7,9,1];
b=ismember(1,a)
c=a(a>5)
d = length(c)