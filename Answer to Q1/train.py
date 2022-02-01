import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import Model
from datetime import datetime
import time

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# MNIST DATA
dataset_dir = "./dataset"
# train data
train_data = torchvision.datasets.MNIST(root=dataset_dir, train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
# test data
test_data = torchvision.datasets.MNIST(root=dataset_dir, train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# Selecting classes 0 and 7
idx = (train_data.targets == 0) | (train_data.targets == 7)
train_data.targets = train_data.targets[idx]
train_data.data = train_data.data[idx]

idx = (test_data.targets == 0) | (test_data.targets == 7)
test_data.targets = test_data.targets[idx]
test_data.data = test_data.data[idx]

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The length of the training dataset is: {}".format(train_data_size))  # 12188
print("The length of the testing dataset is: {}".format(test_data_size))    # 2008

# loading data using "DataLoader"
batch_size = 100
train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=False)

# constructing model
num_instances = 100
num_features = 28
num_bags = 21
model = Model(num_classes=1, num_instances=num_instances, num_features=num_features, num_bins=num_bags, sigma=0.05).to(device)

# loss function
loss_fn = nn.L1Loss().to(device)

# optimizer
learning_rate = 1e-4
weight_decay = 0.0005
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

# setting parameters
# record steps of training
total_train_step = 0
# record steps of testing
total_test_step = 0
# train epoch
epoch = 50

# add tensorboard
metrics_dir = "loss_data"
current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = '{}/step_loss_acc_metrics{}.txt'.format(metrics_dir, current_time)
model_dir = "./model"
writer1 = SummaryWriter("./logs_{}/train".format(current_time))
writer2 = SummaryWriter("./logs_{}/validation".format(current_time))

# save metric

with open(metrics_file,'w') as f_metric_file:
    f_metric_file.write('# Model parameters:\n')
    f_metric_file.write("# Training Data: numer of images: {}\n".format(train_data_size))
    f_metric_file.write("# Validation Data: numer of images: {}\n".format(test_data_size))
    f_metric_file.write('# epoch\ttraining_loss\tvalidation_loss\n')

start_time = time.time()
for i in range(epoch):
    print("--------the {}th epoch of training starts--------".format(i+1))

    # training starts
    model.train()

    total_train_loss = 0
    num_predict = 0

    for data in train_dataloader:
        imgs, targets = data
        # print(imgs.size())      # ([batch_size=100, channel=1, height=28, width=28])
        # print(targets.size())   # 100
        imgs = imgs.to(device)
        targets = targets.to(device)
        new_shape = (1, len(targets))
        targets = targets.view(new_shape)
        outputs = model(imgs)
        # print(outputs.shape, targets.shape)  # [1, num_classes]; [batch_size]
        loss = loss_fn(outputs, targets)

        # optimizer model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        num_predict += targets.size(0)

    train_loss = total_train_loss/len(train_dataloader)
    print("Loss on the train dataset: {}".format(train_loss))
    writer1.add_scalar("loss", train_loss, total_train_step)
    total_train_step += 1

    # testing starts
    model.eval()
    total_test_loss = 0
    num_predict = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            new_shape = (1, len(targets))
            targets = targets.view(new_shape)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()
            num_predict += targets.size(0)

            # accuracy = (outputs.argmax(1) == targets).sum()
            # total_accuracy += accuracy

    test_loss = total_test_loss/len(test_dataloader)
    print("Total loss on the test dataset: {}".format(test_loss))
    # print("Total accuracy on the test dataset: {}".format(total_accuracy/test_data_size))
    writer2.add_scalar("loss", test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    with open(metrics_file, 'a') as f_metric_file:
        f_metric_file.write('%d\t%5.3f\t%5.3f\n' % (i + 1, total_train_loss, total_test_loss))

    # save model
    if (i+1) % 10 == 0:
        model_file = model_dir + "/model_{}.pth".format(i+1)
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state_dict, model_file)
        print("model saved")

    end_time = time.time()
    print("time cost in epoch {} is {}.".format(i+1, end_time-start_time))

model_file = model_dir + "/model_{}.pth".format(i+1)
state_dict = {'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_file)
print("model saved")

writer1.add_graph(model, imgs)

writer1.close()
writer2.close()



