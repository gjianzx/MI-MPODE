function MIfeat = MI_TOOL(X,y,K)

 num_features = size(X,2);
mi_values = zeros(1, num_features);

for i = 1:num_features
    % 计算特征与目标之间的互信息
    mi_values(i) = mi(X(:,i), y, 10);  % 使用相应的互信息函数
end

% 对特征进行排序
[sorted_mi, sorted_indices] = sort(mi_values, 'descend');

% 选择前k个特征
k = round(num_features*K);  % 可以根据需要进行更改
selected_features_indices = sorted_indices(1:k);
MIfeat = X(:, selected_features_indices);

end

function MI = mi(X, Y, numBins)
    % 计算X和Y的联合概率矩阵
%     jointXY = hist3([X, Y], {unique(X), unique(Y)});
%     jointXY = jointXY / sum(jointXY(:));
    set(0, 'DefaultFigureVisible', 'off');    % 不显示图形
    h = histogram2(X, Y, numBins, 'Normalization', 'probability', 'Visible', 'off');
    jointXY = h.Values;

    
    % 计算X和Y的边缘概率
    pX = sum(jointXY, 2);
    pY = sum(jointXY, 1);

    % 计算互信息
    MI = 0;
    for i = 1:length(pX)
        for j = 1:length(pY)
            if jointXY(i, j) > 0
                MI = MI + jointXY(i, j) * log2(jointXY(i, j) / (pX(i) * pY(j)));
            end
        end
    end
end