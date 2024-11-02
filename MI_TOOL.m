function MIfeat = MI_TOOL(X,y,K)

 num_features = size(X,2);
mi_values = zeros(1, num_features);

for i = 1:num_features
    % ����������Ŀ��֮��Ļ���Ϣ
    mi_values(i) = mi(X(:,i), y, 10);  % ʹ����Ӧ�Ļ���Ϣ����
end

% ��������������
[sorted_mi, sorted_indices] = sort(mi_values, 'descend');

% ѡ��ǰk������
k = round(num_features*K);  % ���Ը�����Ҫ���и���
selected_features_indices = sorted_indices(1:k);
MIfeat = X(:, selected_features_indices);

end

function MI = mi(X, Y, numBins)
    % ����X��Y�����ϸ��ʾ���
%     jointXY = hist3([X, Y], {unique(X), unique(Y)});
%     jointXY = jointXY / sum(jointXY(:));
    set(0, 'DefaultFigureVisible', 'off');    % ����ʾͼ��
    h = histogram2(X, Y, numBins, 'Normalization', 'probability', 'Visible', 'off');
    jointXY = h.Values;

    
    % ����X��Y�ı�Ե����
    pX = sum(jointXY, 2);
    pY = sum(jointXY, 1);

    % ���㻥��Ϣ
    MI = 0;
    for i = 1:length(pX)
        for j = 1:length(pY)
            if jointXY(i, j) > 0
                MI = MI + jointXY(i, j) * log2(jointXY(i, j) / (pX(i) * pY(j)));
            end
        end
    end
end