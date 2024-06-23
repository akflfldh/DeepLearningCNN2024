import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
import time

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class InMemoryImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self,root,transform):
        super(InMemoryImageFolder,self).__init__(root,transform)
        self.images=[]
        self.targets=[]

        for idx in range(len(self.samples)):
            path,target=self.samples[idx]
            image=Image.open(path).convert('RGB')
            if self.transform:
                image=self.transform(image)
            self.images.append(image)
            self.targets.append(target)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        return image, target




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 42, 5)
        self.conv4 = nn.Conv2d(42, 130, 5)
        self.conv5 = nn.Conv2d(130, 560, 5)
        #self.conv6 = nn.Conv2d(560, 1704, 5)


        #self.fc1 = nn.Linear(5 * 5 * 16, 120)
        #self.fc1 = nn.Linear(13 * 13 * 16, 900)
        self.fc1 = nn.Linear(4 * 4 * 560, 2300)
        self.fc2 = nn.Linear(2300, 780)
        self.fc3 = nn.Linear(780, 280)
        self.fc4 = nn.Linear(280, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        #x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=1)
        return x



transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('')

    # CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치를 출력합니다:

    print(device)

    batch_size = 4
    # imageFolder로 이미지 읽어들이기
    #trainset = torchvision.datasets.ImageFolder("./mydata/catdog/", transform=transform)
    dataSet = InMemoryImageFolder("./mydata/catdog/", transform=transform)

    datalength=len(dataSet)

    trainsetLength=int(datalength*0.7)
    validationsetLength=int(datalength*0.2)
    testsetlength=datalength-trainsetLength-validationsetLength

    trainSet, validationSet,testSet= random_split(dataSet, [trainsetLength, validationsetLength,testsetlength])



    # 2) 로드한 데이터를 iterator로 변환
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=4,prefetch_factor=2,pin_memory=True)
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=64, shuffle=True, num_workers=4,prefetch_factor=2,pin_memory=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True, num_workers=4,prefetch_factor=2,pin_memory=True)



    # ===============================================================
    # 테스트 시킬 데이터 셋
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 테스트 셋도 마찬가지
    #testset = torchvision.datasets.ImageFolder("./mydata/catdog/", transform=transform)

    #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4,prefetch_factor=2,pin_memory=True)

    classes = ('cat', 'dog')


#==================================================


    net = Net().to(device)


    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer=optim.Adam(net.parameters(),lr=0.0001)

    scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    start_time=time.time()

    for epoch in range(30):  # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



            print("epoch : ", epoch)
            # 통계를 출력합니다.
            running_loss += loss.item()
            print(loss.item())
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        total=0
        correct=0
        for data in validationLoader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            outputs = net(images)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the validation images: {100 * correct // total} %')
        scheduler.step()
    print('Finished Training')

    end_time=time.time()
    elapsed_time=end_time-start_time
    print(elapsed_time)
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)



   # dataiter = iter(testloader)
    #images, labels = next(dataiter)


#=========================================================


    # 이미지를 출력합니다.
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))
    #images=images.to(device)
    #labels=labels.to(device)
    net = Net()
    net.load_state_dict(torch.load('./cifar_net.pth'))
    net.to(device)
    net.eval()

    #outputs = net(images)

    #_, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
     #                             for j in range(1)))

    correct = 0
    total = 0
    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
    with (torch.no_grad()):
        for data in testLoader:
            #images, labels = data
            images,labels=data[0].to(device),data[1].to(device)
            # 신경망에 이미지를 통과시켜 출력을 계산합니다
            outputs = net(images)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')







if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()