%----------------------------------------------------------%
%  EMG Feature Selection and Classification Using          %
%  a Pbest-Guide Binary Particle Swarm Optimization (2019) %
%----------------------------------------------------------%

function PBPSO = gPBPSO(feat,label,opts)
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
Xpbest = zeros(max_Iter,N,dim);
% PBest
Xpb  = X; 
fitP = fit;
Xpbest(1,:,:) = Xpb;
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
MV = zeros(N,dim);
t = 2;  
pcount = 0;
% Iterations
while t <= max_Iter
  CR = 0.9 - 0.9 * (t/max_Iter);
  for i = 1:N
    for d = 1:dim
      r1 = rand();
      r2 = rand();
      % Velocity update (2a)
      VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + c2 * r2 * (Xgb(d) - X(i,d));
      % Velocity limit
      VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
      V(i,d) = VB;
      T = 1/(1 + exp(-10*(V(i,d) - 0.5) ) );
      if T > rand
          X(i,d) = 1;
      else
          X(i,d) = 0;
      end
    end
    % Fitness
    fit(i) = fun(feat,label,X(i,:),opts);
    % Pbest update
    if fit(i) < fitP(i)
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(i);
      pcount = 0;
    else 
      Xpb(i,:) = Xpbest(t-1,i,:); 
      pcount = pcount + 1;
    end
    Xpbest(t,i,:) = Xpb(i,:);   % 保存不好的Xpb
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  %pbest 增强策略
  if pcount >= 2
      pcount = 0;
      for i = 1:N
            R = randperm(N);  R(R == i) = [];
            r1 = R(1); r2 = R(2); r3 = R(3);
            for d = 1:dim
              if Xpb(r1,d) == Xpb(r2,d) 
                diffV = 0;
              else
                diffV = Xpb(r1,d); 
              end
              if diffV == 1
                MV(i,d) = 1;            % MV为变异向量
              else
                MV(i,d) = Xpb(r3,d);
              end
            end
            jrand = randi([1,dim]); 
            for d = 1:dim
              if rand() <= CR  ||  d == jrand 
                New_Xpb(i,d) = MV(i,d);     % 交叉产生的试验向量
              else
                New_Xpb(i,d) = Xpb(i,d);
              end 
            end
      
            fit(i) = fun(feat,label,New_Xpb(i,:),opts);
            if fit(i) < fitP(i)
              New_Xpb(i,:) = X(i,:); 
              fitP(i)  = fit(i);
            end
            % Gbest update
            if fitP(i) < fitG
              Xgb  = New_Xpb(i,:);
              fitG = fitP(i);
            end
      end
  end
  
  curve(t) = fitG; 
%   fprintf('\nIteration %d Best (PSO)= %f',t,curve(t));
  t = t + 1;
end
fprintf('\n Best fitness (PBPSO): %f',fitG);
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
PBPSO.sf = Sf; 
PBPSO.ff = sFeat;
PBPSO.nf = length(Sf);
PBPSO.c  = curve;
PBPSO.f  = feat;
PBPSO.l  = label;
end


