%-------------------------------------------------------------------------%
%  A novel multi-swarm particle swarm optimization for feature selection (2019)  %
%-------------------------------------------------------------------------%

function MSPSO = gMSPSO(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;            % 阈值
c1    = 1.49445;              % cognitive factor
c2    = 1.49445;              % social factor 
w     = 0.729;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity 

% isfield判断输入是否是结构体数组的成员
if isfield(opts,'N'), N = opts.N; end             % 种群数
if isfield(opts,'T'), max_Iter = opts.T; end      % 最大迭代数
if isfield(opts,'c1'), c1 = opts.c1; end          % 学习因子
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'w'), w = opts.w; end             % 权重
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);   %特征的总数当作维度
% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 

fNum = round(N * 0.33);   % 第一个种群
sNum = round(N * 0.33);   % 第二个种群
tNum = N - fNum - sNum;   % 第三个种群

for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N 
  fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
  % Gbest update
  if fit(i) < fitG
    Xgb  = X(i,:); 
    fitG = fit(i);
  end
end
% PBest
Xpb  = X; 
fitP = fit;
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;  
% Iterations
while t <= max_Iter
  % 为三个种群分配个体  首先找出每个子群的pbest和lbest
  [~,f_index] = sort(fit(1:fNum));
  [~,s_index] = sort(fit((fNum + 1):(fNum + sNum)));
  [~,t_index] = sort(fit((fNum + sNum + 1):N));
  f_lbest = X(f_index(1),:);  s_lbest = X(s_index(1),:);  t_lbest = X(t_index(1),:);
  for i = 1:fNum
      for d = 1:dim
        VB = w * V(i,d) + c1 * rand * (Xpb(i,d) - X(i,d)) + c2 * rand * (f_lbest(d) - X(i,d)); 
        VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax; V(i,d) = VB;
      end
      if i == f_index(1)          % 子种群的lbest不参与进化
          X(i,:) = ((f_lbest + s_lbest + t_lbest) ./ 3) .* (1 + normrnd(0,1,1,dim) );
      else
          if rand < 0.6 
              X(i,:) = X(i,:) + V(i,:);
          else
              X(i,:) = Xpb(i,:) .* (1 + normrnd(0,1,1,dim));
          end
      end
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      X(i,:) = XB;
      fit(i) = fun(feat,label,(X(i,:) > thres),opts);
      % Pbest update
      if fit(i) < fitP(i)
        Xpb(i,:) = X(i,:); 
        fitP(i)  = fit(i);
      end
  end
  for i = (fNum + 1):(fNum + sNum)
      for d = 1:dim
        VB = w * V(i,d) + c1 * rand * (Xpb(i,d) - X(i,d)) + c2 * rand * (s_lbest(d) - X(i,d));      
        VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;   V(i,d) = VB;
        X(i,d) = X(i,d) + V(i,d);        
      end
      if i == s_index(1)          % 子种群的lbest不参与进化
          X(i,:) = ((f_lbest + s_lbest + t_lbest) ./ 3) .* (1 + normrnd(0,1,1,dim) );
      else
          if rand < 0.6 
              X(i,:) = X(i,:) + V(i,:);
          else
              X(i,:) = Xpb(i,:) .* (1 + normrnd(0,1,1,dim));
          end
      end      
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      X(i,:) = XB;
      fit(i) = fun(feat,label,(X(i,:) > thres),opts);
      if fit(i) < fitP(i)
        Xpb(i,:) = X(i,:); 
        fitP(i)  = fit(i);
      end
  end
  for i = (fNum + sNum + 1):N
      for d = 1:dim
        VB = w * V(i,d) + c1 * rand * (Xpb(i,d) - X(i,d)) + c2 * rand * (t_lbest(d) - X(i,d)); 
        VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;  V(i,d) = VB;
        X(i,d) = X(i,d) + V(i,d);        
      end
      if i == t_index(1)          % 子种群的lbest不参与进化
          X(i,:) = ((f_lbest + s_lbest + t_lbest) ./ 3) .* (1 + normrnd(0,1,1,dim) );
      else
          if rand < 0.6 
              X(i,:) = X(i,:) + V(i,:);
          else
              X(i,:) = Xpb(i,:) .* (1 + normrnd(0,1,1,dim));
          end
      end      
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      X(i,:) = XB;
      fit(i) = fun(feat,label,(X(i,:) > thres),opts);
      if fit(i) < fitP(i)
        Xpb(i,:) = X(i,:); 
        fitP(i)  = fit(i);
      end
  end
  
  for i = 1:N
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  
  curve(t) = fitG; 
  t = t + 1;
end
fprintf('\n Best fitness (MSPSO): %f',fitG);
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
MSPSO.gb = fitG;
MSPSO.sf = Sf; 
MSPSO.ff = sFeat;
MSPSO.nf = length(Sf);
MSPSO.c  = curve;
MSPSO.f  = feat;
MSPSO.l  = label;
end


