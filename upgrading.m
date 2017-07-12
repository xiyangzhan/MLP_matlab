function net = upgrading(net)

net.w = net.w - net.lr * net.d_w/net.batch_size - net.moment * net.mw;
net.v = net.v - net.lr * net.d_v/net.batch_size - net.moment * net.mv;
net.wb = net.wb - net.lr * sum(net.d_wb,2)/net.batch_size;
net.vb = net.vb - net.lr * sum(net.d_vb,2)/net.batch_size;

net.mw = net.lr * net.d_w/net.batch_size;
net.mv = net.lr * net.d_v/net.batch_size;
end