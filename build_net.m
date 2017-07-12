function net = build_net(Input, Hidden, Output, learning_rate, batch_size, moment)
% initializing param
net.num_input = Input;
net.num_hidden = Hidden;
net.num_output = Output;
net.lr = learning_rate;
net.batch_size = batch_size;
net.moment = moment;

net.w = rand([net.num_hidden, net.num_input]) - 0.5;
net.v = rand([net.num_output, net.num_hidden]) - 0.5;
net.wb = zeros(net.num_hidden,1);
net.vb = zeros(net.num_output,1);
net.mw = zeros(size(net.w));
net.mv = zeros(size(net.v));
end