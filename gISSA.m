%---------------------------------------------------------------------%
%  Improved Salp Swarm Algorithm based on opposition based learning   %
%  and novel local search algorithm for feature selection  (2020)     %
%---------------------------------------------------------------------%

function ISSA = gISSA(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5;

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'thres'), thres = opts.thres; end

fun = @jFitnessFunction;
dim = size(feat,2); 
% Initial  
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end

% ·´ÏòÑ§Ï°
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
% Pre
fit  = zeros(1,N);
fitF = inf;

curve = inf; 
t = 1; 

while t <= max_Iter
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best food update
    if fit(i) < fitF
      Xf   = X(i,:);
      fitF = fit(i); 
    end
  end
	% Compute coefficient, c1 (3.2)
  c1 = 2 * exp(-(4 * t / max_Iter) ^ 2);
  for i = 1:N
      % Leader update
      if i == 1
         for d = 1:dim
            % Coefficient c2 & c3 [0~1]
            c2 = rand();  c3 = rand();
            % Leader update (3.1)
            if c3 >= 0.5 
              X(i,d) = Xf(d) + c1 * ((ub - lb) * c2 + lb);
            else
              X(i,d) = Xf(d) - c1 * ((ub - lb) * c2 + lb);
            end
         end
      % Salp update
      elseif i >= 2
         for d = 1:dim
            % Salp update by following front salp (3.4)
            X(i,d) = (X(i,d) + X(i-1,d)) / 2;
         end
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      X(i,:) = XB;
  end 
  r1 = randi([1,dim]); r2 = randi([1,dim]); r3 = randi([1,dim]); 
  while r1 == r2
      r2 = randi([1,dim]);
  end
  while r2 == r3 || r1 == r3
      r3 = randi([1,dim]);
  end
  sFeats=[r1,r2,r3];
  for j = 1:3
      if (Xf(sFeats(j))>thres) == 1 
          Xf(sFeats(j))=0;
      else
          Xf(sFeats(j))=1;
      end
  end
  TXf = Xf;
  f = fun(feat,label,(TXf(:) > thres),opts);
  if f < fitF
      Xf = TXf;
      curve(t) = f; 
  else
      curve(t) = fitF; 
  end
  t = t + 1;
end
fprintf('\n Best fitness (ISSA): %f',curve(max_Iter));
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xf > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
ISSA.gb = curve(max_Iter);
ISSA.sf = Sf;
ISSA.ff = sFeat; 
ISSA.nf = length(Sf);
ISSA.c  = curve;
ISSA.f  = feat; 
ISSA.l  = label;
end