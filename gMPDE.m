
function MPDE = gMPDE(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;
% CR    = 0.9;  % crossover rate
% F     = 0.5;  % constant factor

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'thres'), thres = opts.thres; end
% if isfield(opts,'CR'), CR = opts.CR; end
% if isfield(opts,'F'), F = opts.F; end

% Function
fun = @jFitnessFunction; 
% Dimension 
dim = size(feat,2);
% Initialize positions 
% Initial
OU = zeros(N,dim);
X = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  

eNum = round(N * 0.2);   % 精英个体总数
iNum = round(N * 0.2);   % 劣质个体总数
oNum = N - eNum - iNum;  % 普通个体总数
mNum = round(oNum * 0.5);
eX = zeros(eNum,dim);
iX = zeros(iNum,dim);
oX = zeros(oNum,dim);
mX = zeros(mNum,dim);

% 反向学习
OX = zeros(N,dim);
a = max(X);
b = min(X);
for i = 1:N
    for d = 1:dim
        OX(i,d) = rand * (a(d) + b(d)) - X(i,d);
    end
    if fun(feat,label,(X(i,:) > thres),opts) > fun(feat,label,(OX(i,:) > thres),opts)
        X(i,:) = OX(i,:);
    end
end
% Fitness
fit  = zeros(1,N); 
fitG = inf; 
for i = 1:N
  fit(i) = fun(feat,label,X(i,:) > thres,opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Pre
U = zeros(N,dim); 
V = zeros(N,dim); 

curve = zeros(1,max_Iter); 
curve(1) = fitG;
t = 2; 
FL = zeros(max_Iter,dim);
FL(1,:) = rand(1,dim)*(0.8-0.5)+0.5;
F = zeros(max_Iter,N); F(1,:) = 0.5;
CR = zeros(max_Iter,N); CR(1,:) = 0.9;

while t <= max_Iter
    
    FL(t,:) = 3.0 .* FL(t-1,:) .* (1 - FL(t-1,:));
    % 种群个体排序
    [~,index] = sort(fit);
    for i = 1:eNum                  % 精英个体  应集中子种群开发最优解
        k = randi([1 eNum]); 
        if fit(index(i)) <= fit(index(k))    % 可能为最优解，方差较小产生较小扰动
            miu = 0; %0.3;
            sigma = 1; % 0.1;
        else
            miu = 0; %0.4;
            sigma = exp( (fit(index(k)) - fit(index(i))) / ( abs(fit(index(i)))+ realmin) ); %0.13; 
        end
        V(index(i),:) = X(index(i),:) .* (1 + normrnd(miu,sigma,1,dim));
        U(index(i),:) = V(index(i),:);     
        eX(i,:) = X(index(i),:);
    end
    
    for i = (eNum + 1) : (eNum +oNum)   % 普通个体  集中子种群之间的通信
        oX(i,:) = X(index(i),:);
        r1 = randi([1 size(eX,1)]);
        r2 = randi([(eNum + 1)  (eNum +oNum)]);
        if rand < 0.1
            F(t,i) = 0.1 + rand * 0.9;
            CR(t,i) = rand;
        else
            F(t,i) = F(t-1,i);
            CR(t,i) = CR(t-1,i);
        end

        V(index(i),:) = X(index(i),:) + F(t,i) .* ( eX(r1,:) - X(index(i),:))...
                                      + (1-F(t,i))  .* (X(index(r2),:) - X(index(i),:));                    
        rnbr = randi([1,dim]);                          
        for d = 1:dim
          if rand() <= CR(t,i) || d == rnbr 
            U(index(i),d) = V(index(i),d);
          else
            U(index(i),d) = X(index(i),d);
          end
        end                                                            
    end
    m = randperm(size(oX,1),mNum);
   
    for i = (eNum + oNum + 1) : N                    % 劣质个体  集中子种群的探索
        iX(i,:) = X(index(i),:);
        r3 = m(randperm(length(m),1));
        V(index(i),:) = X(index(i),:) + FL(t,:) .* ( oX(r3,:) - X(index(i),:) );
        rnbr = randi([1,dim]);
        if rand < 0.3
            CR(t,i) = rand;
        else
            CR(t,i) = CR(t-1,i);
        end
        for d = 1:dim
          if rand() <= CR(t,i) || d == rnbr 
            U(index(i),d) = V(index(i),d);
          else
            U(index(i),d) = X(index(i),d);
          end
        end        
    end
    
  for i = 1:N
    XB = U(i,:); XB(XB > ub) = ub; XB(XB < lb) = ub;
    U(i,:) = XB;
    if rand < 0.1
        a = max(X);b = min(X);
        OU(i,:) = rand(1,dim) .* (a + b) - U(i,d);
    end
    % Fitness
    Fnew = fun(feat,label,(U(i,:) > thres),opts);
    OFnew = fun(feat,label,(OU(i,:) > thres),opts);
    temp = fit(i);
    % Selection
    if Fnew < temp
      X(i,:) = U(i,:);
      fit(i) = Fnew;
      temp = Fnew;
    end
    if OFnew < temp
        X(i,:) = OU(i,:);
        fit(i) = OFnew;
    end
    % Best update
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  t = t + 1;
end
fprintf('\n Best fitness (MPODE): %f',fitG);
% Select features based on selected index
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
MPDE.gb = fitG;
MPDE.sf = Sf; 
MPDE.ff = sFeat; 
MPDE.nf = length(Sf);
MPDE.c  = curve;
MPDE.f  = feat;
MPDE.l  = label;
end

