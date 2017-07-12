function [output_index, label_index, accurate] = test(net, batch_x, batch_y)
net = forward(net, batch_x, batch_y);
output = net.o_o; % 提取输出
[~, output_index] = max(output); % 获取输出最大的节点位置
[~, label_index] = max(batch_y'); % 获取真实数字
output_index = output_index -1;  % 将1-10 变为 0-9
label_index = label_index - 1;   % 将1-10 变为 0-9

num = length(find(output_index == label_index)); % 判断输出最大值位置与标签对应数字相同的个数
accurate = num/size(batch_x,1); % 正确识别数除以总数
end