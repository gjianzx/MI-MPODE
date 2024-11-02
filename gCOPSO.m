%--------------------------------------------------------------%
%  A conditional opposition-based particle swarm optimisation  %
%  for feature selection (2022)                                %     
%--------------------------------------------------------------%

function COPSO = gCOPSO(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;            % 阈值
c1    = 2;              % cognitive factor
c2    = 2;              % social factor 
w     = 0.9;            % inertia weight
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
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end
% 反向学习
OX = zeros(N,dim);
a = max(X);
b = min(X);
for i = 1:N
    for d = 1:dim
        OX(i,d) = a(d) + b(d) - X(i,d);
    end
    if fun(feat,label,X(i,:) > thres,opts) > fun(feat,label,OX(i,:) > thres,opts)
        X(i,:) = OX(i,:);
    end
end

% Fitness
fit  = zeros(max_Iter,N); 
XGbest = zeros(max_Iter,dim);
fitG = inf;
for i = 1:N 
  fit(1,i) = fun(feat,label,(X(i,:) > thres),opts); 
  % Gbest update
  if fit(1,i) < fitG
    Xgb  = X(i,:); 
    fitG = fit(1,i);
  end
end
XGbest(1,:) = Xgb;

% PBest
Xpb  = X; 
fitP = fit(1,:);
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;  
% Iterations
while t <= max_Iter
  Xm = mean(X,1);         % 当前种群的平均位置
  Xmg = mean(XGbest,1);   % 迭代过程中全局最优粒子的平均位置
  for i = 1:N
    if fit(t,i)<= fit(t-1,i)    % 探索阶段
      w = 0.5 + 0.5*rand;
      for d = 1:dim
          r1 = rand();r2 = rand();
          VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + c2 * r2 * (Xm(d) - X(i,d));
          S = 1/(1+exp(-10*VB-0.5));
          % Velocity limit
          VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
          V(i,d) = VB;
          % Position update (2b)
          if S > rand
              X(i,d) = 1;
          else
              X(i,d) = 0;
          end          
      end 
    else                        % 开发阶段
      w = 0.8 * rand;
      for d = 1:dim
          r1 = rand();r2 = rand();
          % Velocity update (2a)
          VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + c2 * r2 * (Xmg(d) - X(i,d));
          S = 1/(1+exp(-10*VB-0.5));
          % Velocity limit
          VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
          V(i,d) = VB;
          % Position update (2b)
          if S > rand
              X(i,d) = 1;
          else
              X(i,d) = 0;
          end 
      end         
    end
      
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
    % Fitness
    fit(t,i) = fun(feat,label,X(i,:),opts);
    % Pbest update
    if fit(t,i) < fitP(i)
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(t,i);
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  % 每次迭代结束，OBL用于gbest
  a = max(X); b = min(X);
  OXgb = a + b - Xgb;
  if fun(feat,label,Xgb,opts) > fun(feat,label,OXgb,opts)
      Xgb = OXgb;
  end
  XGbest(t,:) = Xgb;
  
  curve(t) = fitG; 
%   fprintf('\nIteration %d Best (PSO)= %f',t,curve(t));
  t = t + 1;
end
fprintf('\n Best fitness (COPSO): %f',fitG);
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
COPSO.gb = fitG;
COPSO.sf = Sf; 
COPSO.ff = sFeat;
COPSO.nf = length(Sf);
COPSO.c  = curve;
COPSO.f  = feat;
COPSO.l  = label;
end


