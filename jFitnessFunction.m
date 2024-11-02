% Fitness Function KNN (9/12/2020)

% == == == == == == == == Input == == == == == == == == %
% feat  : 特征                                          %
% label : 标签(类别)                                    %
% X     : 连续编码方式，>0.5即选择特征为1，反之不选此特征 %
% opts  : 进化算法的参数结构体                           %
% == == == == == == == == == == == == == == == == == == %

% == == == == == == == == Onput == == == == == == == == %
% cost  : 函数评估的适应值                               %
% == == == == == == == == == == == == == == == == == == %

function cost = jFitnessFunction(feat,label,X,opts)
% Default of [alpha; beta]
% ws = [0.99; 0.01];
ws = [0.95; 0.05];
if isfield(opts,'ws'), ws = opts.ws; end

% Check if any feature exist
if sum(X == 1) == 0
  cost = 1;
else
  % Error rate
  error    = jwrapper_KNN(feat(:,X == 1),label,opts);
  % Number of selected features    选择的特征的数量
  num_feat = sum(X == 1);
  % Total number of features       特征总数
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  % Cost function 
  cost     = alpha * error + beta * (num_feat / max_feat); 
end
end

%---------------------Call Functions----------------------
function error = jwrapper_KNN(sFeat,label,opts)
if isfield(opts,'k'), k = opts.k; end               % k为类别数，k=5
if isfield(opts,'Model'), Model = opts.Model; end   % Model是交叉验证得到的结构体

% Define training & validation sets
trainIdx = Model.training;    testIdx = Model.test;
xtrain   = sFeat(trainIdx,:); ytrain  = label(trainIdx);
xvalid   = sFeat(testIdx,:);  yvalid  = label(testIdx);
% Training model
My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); %fitcknn构造KNN分类器
% Prediction
pred     = predict(My_Model,xvalid);  %predict预测值，My_Model是一个训练好的模型，xvalid是一个输入数据，预测xvalid对应的输出值。
% Accuracy
Acc      = sum(pred == yvalid) / length(yvalid);
% Error rate
error    = 1 - Acc; 
end











