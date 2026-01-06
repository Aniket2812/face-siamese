import torch
import torch.nn as nn
from layers import L1Dist
import torch.nn.functional as F

#embedding layer
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        #first block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #second block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #third block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #final embedding block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)

        #fully connected linear layer
        self.fc = nn.Linear(256 * 5 * 5, 4096)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1) #flatten
        x = torch.sigmoid(self.fc(x)) #dense (4096, sigmoid)

        return x
    
#siamese model
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()

        self.embedding = embedding_model
        self.l1 = L1Dist()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        #embeddings
        emb1 = self.embedding(img1)
        emb2 = self.embedding(img2)

        #L1 distance
        distance = self.l1(emb1, emb2)

        #classifier
        output = self.classifier(distance)

        return output