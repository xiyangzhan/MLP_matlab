load mnist %������д�����ݼ�
learning_rate = 0.01; %ѧϰ��
moment = 0.9; %����
num_iter = 10; %��������
batch_size = 32; 
num_I = 784; %�����ڵ����
num_H = 200; %������ڵ����
num_O = 10; %�����ڵ����
train_x = double(train_x)/255; % ��һ��ѵ�����������ݣ���ת��Ϊ������
train_y = double(train_y);  %ѵ������ǩ
test_x = double(test_x)/255; %��һ�����Լ��������ݣ���һ��������ѵ������
test_y = double(test_y); %���Լ���ǩ

net = build_net(num_I, num_H, num_O, learning_rate, batch_size, moment); %��������
num_temp = randperm(length(train_y)); %����������˳��

L = []; %�����洢���
for i = 1:num_iter
    for j = 1:length(num_temp)/batch_size
        batch_x = train_x(num_temp((j-1)*batch_size+1:j*batch_size),:);  %��train_x����ȡbatch_size������
        batch_y = train_y(num_temp((j-1)*batch_size+1:j*batch_size),:);  %��train_y����ȡ��Ӧ��������ǩ
        net = forward(net, batch_x, batch_y); %ǰ�򴫲�
        net = backward(net); %���򴫲�
        net = upgrading(net); %Ȩֵ����
        L = [L, net.loss];  %�洢��ʧ
        if mod(j,500)==0
            [number, label, accurate] = test(net, test_x, test_y); %���Ժ��������ز�������������֣���ǩ��Ӧ�����֣� ���Լ�����
            disp(['������',num2str(j+(i-1)*60000/batch_size),'��,ѵ������ʧΪ��'...
                ,num2str(L(end)),'. ���Լ�����Ϊ��',num2str(accurate),'.']) %��ʾ
        end
    end
end
plot(L); %����ѵ����ʧ����


