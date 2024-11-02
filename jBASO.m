%----------------------------------------------------%
%  Binary atom search optimisation approaches for    %
%  feature selection  (2020)                         %
%----------------------------------------------------%

function BASO = jBASO(feat,label,opts)

% Parameters
alpha = 50;    % depth weight
beta  = 0.2;   % multiplier weight

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'beta'), beta = opts.beta; end 
if isfield(opts,'thres'), thres = opts.thres; end 

fun  = @jFitnessFunction; 
dim  = size(feat,2); 
X    = jInitialPopulation(N,dim); 
V    = zeros(N,dim); 
Vmax = 6;  
for i = 1:N
  for d = 1:dim
    V(i,d) = -Vmax + (Vmax - (-Vmax)) * rand();
  end
end
fit  = zeros(1,N); 
fitG = inf; 
A    = zeros(N,dim);
eps  = 0.001;

curve = inf; 
t = 1; 
%---Iteration start--------------------------------------------------
while t <= max_Iter
  for i = 1:N
    fit(i) = fun(feat,label,X(i,:),opts);
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end
  fitB       = min(fit);
  fitW       = max(fit); 
  Kbest      = ceil(N - (N - 2) * sqrt(t / max_Iter));   
  M          = exp(-(fit - fitB) ./ (fitW - fitB));
  M          = M ./ sum(M);
  [~, idx_M] = sort(M,'descend');
  G          = exp(-20 * t / max_Iter);
  E          = zeros(N,dim); 
  for i = 1:N      
    XK(1,:)    = sum(X(idx_M(1:Kbest), :), 1) / Kbest;
    scale_dist = norm(X(i,:) - XK(1,:), 2);   
    for ii = 1:Kbest
      j    = idx_M(ii);
      Po   = jLJPotential(X(i,:),X(j,:),t,max_Iter,scale_dist);
      dist = norm(X(i,:) - X(j,:), 2);
      for d = 1:dim
        E(i,d) = E(i,d) + rand() * Po * ((X(j,d) - X(i,d)) / ...
          (dist + eps));
      end
    end
    for d = 1:dim
      E(i,d) = alpha * E(i,d) + beta * (Xgb(d) - X(i,d));
      A(i,d) = E(i,d) / M(i); 
    end
  end
  for i = 1:N
    for d = 1:dim
      Acce   = A(i,d) * G;
      VB     = rand() * V(i,d) + Acce;
      VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
      V(i,d) = VB;
      S      = 1 / (1 + exp(-2 * V(i,d)));
      K1     = (1/3) * rand();
      K2     = (1/3) + (1/3) * rand();
      if S <= K1
        X(i,d) = 0;
      elseif S > K1 && S <= K2
        X(i,d) = 1;
      else
        X(i,d) = Xgb(d); 
      end
    end
  end
  curve(t) = fitG; 
  
  t = t + 1;
end
fprintf('\n Best fitness (BASO)= %f',fitG);
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos(Xgb == 1);
sFeat = feat(:,Sf); 
% Store results
BASO.gb = fitG;
BASO.sf = Sf; 
BASO.ff = sFeat; 
BASO.nf = length(Sf); 
BASO.c  = curve; 
BASO.f  = feat; 
BASO.l  = label;
end


function Potential = jLJPotential(X1,X2,t,max_Iter,dim)
h0   = 1.1;
u    = 1.24; 
r    = norm(X1 - X2, 2);
n    = (1 - (t - 1) / max_Iter) .^ 3;
g    = 0.1 * sin((pi / 2) * (t / max_Iter));
Hmin = h0 + g; 
Hmax = u;
if r / dim < Hmin
  H = Hmin;
elseif r / dim > Hmax
  H = Hmax;  
else
  H = r / dim;
end           
Potential = n * (12 * (-H) ^ (-13) - 6 * (-H) ^ (-7)); 
end


function X = jInitialPopulation(N,dim)
X = zeros(N,dim);
for i = 1:N
  for d = 1:dim
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
end

