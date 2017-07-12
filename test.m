function [output_index, label_index, accurate] = test(net, batch_x, batch_y)
net = forward(net, batch_x, batch_y);
output = net.o_o; % ��ȡ���
[~, output_index] = max(output); % ��ȡ������Ľڵ�λ��
[~, label_index] = max(batch_y'); % ��ȡ��ʵ����
output_index = output_index -1;  % ��1-10 ��Ϊ 0-9
label_index = label_index - 1;   % ��1-10 ��Ϊ 0-9

num = length(find(output_index == label_index)); % �ж�������ֵλ�����ǩ��Ӧ������ͬ�ĸ���
accurate = num/size(batch_x,1); % ��ȷʶ������������
end