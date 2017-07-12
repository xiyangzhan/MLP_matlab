function net = backward(net)

net.d_o = net.loss_temp; % 最终误差 [10, 32]
net.d_oi = net.d_o .* net.o_o .* (1 - net.o_o); % 输出层的输入的误差 [10, 32]
net.d_h = net.v' * net.d_oi ; % 隐含层输出的误差 [200, 32]
net.d_hi = net.d_h .* net.h_o .* (1 - net.h_o); % 隐含层的输入的误差 [200, 32]
%net.d_hi = net.d_h .* (net.h_o > 0); % 使用ReLU作为激活函数

net.d_w = net.d_hi * net.x'; % 参数w的误差[200, 784]
net.d_v = net.d_oi * net.h_o'; % 参数v的误差 [10, 200]
net.d_wb = net.d_hi; % 偏置wb的误差[200, 32]
net.d_vb = net.d_oi; % 偏置vb的误差 [10, 32]
end