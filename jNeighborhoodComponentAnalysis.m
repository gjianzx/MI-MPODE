function NCA = jNeighborhoodComponentAnalysis(feat,label,K)

% num_feat = round(size(feat,2)*K); 

% if isfield(opts,'Nf'), num_feat = opts.Nf; end          % opts.Nf 选择特征的数量

% Perform NCA
model    = fscnca(feat,label);
% Weight
weight   = model.FeatureWeights; 
% 权重归一化处理
% W = (weight - min(weight) )./(max(weight) - min(weight) + realmin);
num_feat = length(W(W > K));

% Higher weight better features
[~, idx] = sort(weight,'descend');
% Select features based on selected index
Sf       = idx(1:num_feat)';
sFeat    = feat(:,Sf); 
% Store results
NCA.sf = Sf; 
NCA.ff = sFeat;
NCA.nf = num_feat; 
NCA.f  = feat;
NCA.l  = label;
NCA.s  = weight;
end





      