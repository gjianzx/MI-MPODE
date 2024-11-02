%----------------------------------------------------%
%  A novel three layer particle swarm optimization   %
%  for feature selection (2021)                      %
%----------------------------------------------------%

function TLPSO = gTLPSO(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;            % ��ֵ
c1    = 2;              % cognitive factor
c2    = 2;              % social factor 
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity 

% isfield�ж������Ƿ��ǽṹ������ĳ�Ա
if isfield(opts,'N'), N = opts.N; end             % ��Ⱥ��
if isfield(opts,'T'), max_Iter = opts.T; end      % ��������
if isfield(opts,'c1'), c1 = opts.c1; end          % ѧϰ����
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'w'), w = opts.w; end             % Ȩ��
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2);   %��������������ά��
% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 

eNum = round(N * 0.3);   % ��Ӣ��������
oNum = round(N * 0.3);   % ��ͨ��������
iNum = N - eNum - oNum;  % ���ʸ�������

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
  % ��Ⱥ����: elite  ordinary inferior
  [~,index] = sort(fit); 
  sigma = 0.3 - 0.2 * (t/max_Iter);  
  for i = 1:eNum                  % ��Ӣ����
      for d = 1:dim
          miu=(X(index(i),d) + X(index(1),d))/2;
          X(i,d) = normrnd(miu,sigma*sigma);
      end        
  end
  % Ϊ��ͨ��������ṩ�����������
  p = randi([1,eNum]); q = randi([1,eNum]);
  while (p == q) && (fit(index(p)) < fit(index(q))) 
      q = randi([1 eNum]);
  end
  
  for i = (eNum + 1):(eNum +oNum)        % ��ͨ����
      for d = 1:dim
          r1 = rand();r2 = rand();r3 = rand();
          VB = r1 * V(i,d) + r2 * (X(p,d) - X(i,d)) + 0.4 * r3 * (X(q,d) - X(i,d));
          VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
          V(i,d) = VB;
          X(i,d) = X(i,d) + V(i,d);
      end
  end
  % �Ӿ�Ӣ�������ͨ���������ѡ����������
  L1 = randi([1,eNum]); L2 = randi([(eNum + 1), (eNum +oNum)]);
  for i = (eNum + oNum + 1):N          % ���ʸ���
      for d = 1:dim
          r1 = rand();r2 = rand();r3 = rand();
          VB = r1 * V(i,d) + r2 * (X(L1,d) - X(i,d)) + 0.4 * r3 * (X(L2,d) - X(i,d));
          VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
          V(i,d) = VB;
          X(i,d) = X(i,d) + V(i,d);
      end
  end
  for i = 1:N
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Pbest update
    if fit(i) < fitP(i)
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(i);
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  
  curve(t) = fitG; 
  t = t + 1;
  
end
fprintf('\n Best fitness (TLPSO): %f',fitG);
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
TLPSO.gb = fitG;
TLPSO.sf = Sf; 
TLPSO.ff = sFeat;
TLPSO.nf = length(Sf);
TLPSO.c  = curve;
TLPSO.f  = feat;
TLPSO.l  = label;
end


