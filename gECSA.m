%--------------------------------------------------------------%
%  Enhanced Crow Search Algorithm for Feature Selection (2020) %
%--------------------------------------------------------------%

function ECSA = gECSA(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
AP    = 0.1;   % awareness probability
fl    = 1.5;   % flight length

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'AP'), AP = opts.AP; end 
if isfield(opts,'fl'), fl = opts.fl; end 
if isfield(opts,'thres'), thres = opts.thres; end 

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim); 
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
  % Global update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Save memory
fitM = fit;
Xm   = X;
% Pre
Xnew = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;

% Iteration
while t <= max_Iter
    
    [~,index] = sort(fit);
    [~,rank] = sort(index);
	for i = 1:N
        DAP = 0.1 + (0.8-0.1) *(rank(i)/N);
        % Random select 1 memory crow to follow
        if i > 2 && i < (N-1)
            k = randi([i-2,i+2]);
        elseif i == 1
            A = [N-1,N,1,2,3];
            k = A(randi(length(A),1));
        elseif i == 2
            A = [N,1,2,3,4];
            k = A(randi(length(A),1));            
        elseif i == N
            A = [N-2,N-1,N,1,2];
            k = A(randi(length(A),1));            
        elseif i == (N-1)
            A = [N-3,N-2,N-1,N,1];
            k = A(randi(length(A),1));            
        end
        % Awareness of crow m (2)
        if rand() >= DAP    
          for d = 1:dim
            % Crow m does not know it has been followed (1)
            Xnew(i,d) = X(i,d) + fl * (Xm(k,d) - X(i,d));
            if (1 / (1 + exp(-Xnew(i,d))) ) >= 0.5
                Xnew(i,d) = 1;
            else
                Xnew(i,d) = 0;
            end
          end
        else
            for d = 1:dim
                c1 = 2 * exp(-4 * t / max_Iter);
                c2 = rand();
                if rand < 0.5
                   Xnew(i,d) =  Xgb(d) + c1 * c2;
                else
                   Xnew(i,d) =  Xgb(d) - c1 * c2; 
                end
                if (1 / (1 + exp(-Xnew(i,d)))) >= 0.5
                    Xnew(i,d) = 1;
                else
                    Xnew(i,d) = 0;
                end                
            end
        end
  end
  
  % Fitness
  for i = 1:N
    Fnew = fun(feat,label,Xnew(i,:),opts); 
    % Check feasibility
    if all(Xnew(i,:) >= lb) && all(Xnew(i,:) <= ub)
      % Update crow
      X(i,:) = Xnew(i,:);
      fit(i) = Fnew;
      % Memory update (5)
      if fit(i) < fitM(i)
        Xm(i,:) = X(i,:);
        fitM(i) = fit(i);
      end
      % Global update
      if fitM(i) < fitG
        fitG = fitM(i);
        Xgb  = Xm(i,:);
      end
    end
  end
  curve(t) = fitG; 
  t = t + 1;
end
fprintf('\n Best fitness (ECSA): %f',fitG);
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
ECSA.gb = fitG;
ECSA.sf = Sf; 
ECSA.ff = sFeat;
ECSA.nf = length(Sf); 
ECSA.c  = curve; 
ECSA.f  = feat;
ECSA.l  = label;
end

