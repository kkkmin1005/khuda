'''

Code written by Hayoung Lee.
Contact email: lhayoung9@khu.ac.kr (Note: I may not check my email regularly due to my mistake.)

Training conducted using TPU v2 on Google Colaboratory.

'''

import os
from google.colab import drive

# Mount Google Drive

drive.mount('/content/drive')

# Set project folder path

project_folder = '/content/drive/MyDrive/Project3'

# Initialize lists to store image paths and labels

image = []
label = []

# Traverse through the project folder to collect image paths and corresponding labels

for subdir, _, files in os.walk(project_folder):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(subdir, file)
            image.append(image_path)
            
            label_name = os.path.basename(subdir)
            label.append(label_name)
            

from torch.utils.data import DataLoader
from Preprocessing import CustomDataset
from sklearn.model_selection import train_test_split 

BATCH_SIZE = 128

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size = 0.33, random_state = 425)

# Create custom datasets and dataloaders

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


'''

Declaration of Model, Optimizer, etc.

1) Epoch: 100
2) Batch size: 128
    - Due to the small size of the dataset, batch size was increased based on professor's advice.
3) Loss Function: CrossEntropy
4) Optimizer: Adam with Learning rate 0.01

'''


import time
import torch
import torch.nn as nn

from Model import Recognizer

EPOCH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer

MODEL = Recognizer().to(DEVICE)
LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr = 0.001)


'''

Function Definitions

1) def compute_accuracy_and_loss(device, model, data_loader):
    - Input: device, model, data_loader
    - Output: loss / example_num, correct_num / example_num * 100
        - 'loss / example_num' represents the average loss.
        - 'correct_num / example_num * 100' represents the accuracy percentage.

2) def save_weight(model, path):
    - This function saves the model's weights at the specified path.
    
'''


def compute_accuracy_and_loss(device, model, data_loader):
    loss, example_num, correct_num = 0, 0, 0
    
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.to(device)
        probability = model(image)
        
        #Calculate loss using CrossEntropy
    
        loss += LOSS(probability, label)
        
        #Calculate accuracy
        
        _, true_index = torch.max(label, 1)
        _, predict_index = torch.max(probability, 1)
        
        example_num += true_index.size(0)
        correct_num += (true_index == predict_index).sum
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(data_loader):03d} |'
               f'Loss: {loss:03f}')
        
    return loss/example_num, correct_num/example_num*100


def save_weight(model, path):
    torch.save(model.state_dict(), path)


'''

Visualizing model architecture by using tensorboard Library

'''


from torch.utils.tensorboard import SummaryWriter

image_for_visualization, label_for_visualization = train_dataset[0]

writer = SummaryWriter()
writer.add_graph(MODEL, image_for_visualization.unsqueeze(0))


'''

Training

'''


start_time = time.time()

for epoch in range(EPOCH):
    MODEL.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        probability = MODEL(image)
        
        loss = LOSS(probability, label)
        
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch: {batch_idx:03d}/{len(train_loader):03d} |'
               f'Loss: {loss:03f}')
        
    MODEL.eval()
    with torch.no_grad():
        train_loss, train_acc = compute_accuracy_and_loss(DEVICE, MODEL, train_loader)
        test_loss, test_acc = compute_accuracy_and_loss(DEVICE, MODEL, test_loader)
        
        # Add scalars to tensorboard for visualization
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        writer.flush()
    
    # Save model weights every 10 epochs
    
    if epoch%10 == 0:
        save_weight(MODEL.VGG19, f"/content/drive/MyDrive/VGG19_{epoch}.pth")
        save_weight(MODEL.ArcFace, f"/content/drive/MyDrive/ArcFace_{epoch}.pth")
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs

# 학습을 100 에포크 만큼 진행하며 각 배치에서 모델에 이미지를 입력하고
# 손실을 계산한 후 역전파를 통해 모델의 가중치를 업데이트합니다.
# 매 에포크가 끝날때 마다 loss와 정확도를 기록합니다
# 매 10회의 에포크마다 가중치를 저장합니다.