function net = forward(net, batch_x, batch_y)
net.x = batch_x';
net.y = batch_y';

% L1
net.h = net.w * net.x + repmat(net.wb,1,size(net.x,2));
net.h_o = sigmoid(net.h);
% L2
net.o = net.v * net.h_o + repmat(net.vb,1,size(net.x,2));
net.o_o = sigmoid(net.o);

% loss
net.loss_temp = net.o_o - net.y;
net.loss = sum(sum(1/2*(net.loss_temp).^2))/size(net.x,2);
end