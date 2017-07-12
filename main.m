load mnist %载入手写体数据集
learning_rate = 0.01; %学习率
moment = 0.9; %动量
num_iter = 10; %迭代次数
batch_size = 32; 
num_I = 784; %输入层节点个数
num_H = 200; %隐含层节点个数
num_O = 10; %输出层节点个数
train_x = double(train_x)/255; % 归一化训练集输入数据，并转变为浮点型
train_y = double(train_y);  %训练集标签
test_x = double(test_x)/255; %归一化测试集输入数据，归一化可提升训练精度
test_y = double(test_y); %测试集标签

net = build_net(num_I, num_H, num_O, learning_rate, batch_size, moment); %建立网络
num_temp = randperm(length(train_y)); %打乱样本的顺序

L = []; %用来存储误差
for i = 1:num_iter
    for j = 1:length(num_temp)/batch_size
        batch_x = train_x(num_temp((j-1)*batch_size+1:j*batch_size),:);  %从train_x中提取batch_size个样本
        batch_y = train_y(num_temp((j-1)*batch_size+1:j*batch_size),:);  %从train_y中提取对应的样本标签
        net = forward(net, batch_x, batch_y); %前向传播
        net = backward(net); %反向传播
        net = upgrading(net); %权值更新
        L = [L, net.loss];  %存储损失
        if mod(j,500)==0
            [number, label, accurate] = test(net, test_x, test_y); %测试函数，返回测试数据输出数字，标签对应的数字， 测试集精度
            disp(['迭代第',num2str(j+(i-1)*60000/batch_size),'次,训练集损失为：'...
                ,num2str(L(end)),'. 测试集精度为：',num2str(accurate),'.']) %显示
        end
    end
end
plot(L); %画出训练损失曲线


