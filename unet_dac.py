import torch
import torch.nn as nn
# from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
from config import NUM_CLASSES, MODEL_L, MODEL_K, NUM_MICS

class UnetDAC(nn.Module):
    def __init__(self, L: int = MODEL_L, K: int = MODEL_K, M: int = NUM_MICS, num_classes: int = NUM_CLASSES, dropout_probability: float = 0.25):
        super().__init__()
        self.L = L
        self.K = K
        self.M = M
        self.num_classes = num_classes
        self.input_shape = (L, K, 2 * (M - 1))
        self.dropout_probability = dropout_probability
        self.dropout = nn.Dropout(p=dropout_probability)
        self.instancenorm2d = nn.InstanceNorm2d(2 * (M - 1), affine=False)

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 64x14x256x96 (Batch_size=64 x MicsFactor=14 x FreqBins=256 x Frames~96)
        self.e11 = nn.Conv2d(2 * (M - 1), 16, kernel_size=3, padding=1) # output: 64x16x256x96
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # output: 64x16x256x96
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x16x128x48

        # input: 284x284x64
        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # output: 64x32x128x48
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # output: 64x32x128x48
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x32x64x24

        # input: 140x140x128
        self.e31 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: 64x64x64x24
        self.e32 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 64x64x64x24
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x64x32x12

        # input: 68x68x256
        self.e41 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 64x128x32x12
        self.e42 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 64x128x32x12
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 64x128x16x6

        # input: 32x32x512
        self.e51 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 64x256x16x6
        self.e52 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 64x256x16x6


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 64x128x32x12
        # Cat with 64x128x32x12 --> 64x256x32x12
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(16, num_classes, kernel_size=1)

        nn.init.xavier_uniform_(self.e11.weight)  
        nn.init.xavier_uniform_(self.e12.weight)  
        nn.init.xavier_uniform_(self.e21.weight)  
        nn.init.xavier_uniform_(self.e22.weight)  
        nn.init.xavier_uniform_(self.e31.weight)  
        nn.init.xavier_uniform_(self.e32.weight)  
        nn.init.xavier_uniform_(self.e41.weight)  
        nn.init.xavier_uniform_(self.e42.weight)  
        nn.init.xavier_uniform_(self.e51.weight)  
        nn.init.xavier_uniform_(self.e52.weight)  
        nn.init.xavier_uniform_(self.d11.weight)  
        nn.init.xavier_uniform_(self.d12.weight)  
        nn.init.xavier_uniform_(self.d21.weight)  
        nn.init.xavier_uniform_(self.d22.weight)  
        nn.init.xavier_uniform_(self.d31.weight)  
        nn.init.xavier_uniform_(self.d32.weight)  
        nn.init.xavier_uniform_(self.d41.weight)  
        nn.init.xavier_uniform_(self.d42.weight)  
        nn.init.xavier_uniform_(self.outconv.weight)  
        nn.init.xavier_uniform_(self.upconv1.weight)  
        nn.init.xavier_uniform_(self.upconv2.weight)  
        nn.init.xavier_uniform_(self.upconv3.weight)  

    def forward(self, data):
        # Encoder
        data = self.instancenorm2d(data)
        data = self.dropout(F.elu(self.e11(data)))
        xe12 = self.dropout(F.elu(self.e12(data)))
        data = self.pool1(xe12)

        data = self.dropout(F.elu(self.e21(data)))
        xe22 = self.dropout(F.elu(self.e22(data)))
        data = self.pool2(xe22)

        data = self.dropout(F.elu(self.e31(data)))
        xe32 = self.dropout(F.elu(self.e32(data)))
        data = self.pool3(xe32)

        data = self.dropout(F.elu(self.e41(data)))
        xe42 = self.dropout(F.elu(self.e42(data)))
        data = self.pool4(xe42)

        data = self.dropout(F.elu(self.e51(data)))
        data = self.dropout(F.elu(self.e52(data)))
        
        # Decoder
        data = self.upconv1(data) # 
        data = torch.cat([data, xe42], dim=1) # XXX: Changed from dim=1 in all places.
        """
        xu1.shape
        torch.Size([128, 32, 2])
        xe42.shape
        torch.Size([128, 32, 2])

        xu11.shape
        dim=1: torch.Size([128, 64, 2])
        dim=0: torch.Size([256, 32, 2])
        """
        data = self.dropout(F.elu(self.d11(data)))
        data = self.dropout(F.elu(self.d12(data)))

        data = self.upconv2(data)
        data = torch.cat([data, xe32], dim=1)
        data = self.dropout(F.elu(self.d21(data)))
        data = self.dropout(F.elu(self.d22(data)))

        data = self.upconv3(data)
        data = torch.cat([data, xe22], dim=1)
        data = self.dropout(F.elu(self.d31(data)))
        data = self.dropout(F.elu(self.d32(data)))

        data = self.upconv4(data)
        data = torch.cat([data, xe12], dim=1)
        data = self.dropout(F.elu(self.d41(data)))
        data = self.dropout(F.elu(self.d42(data)))

        # Output layer
        data = self.outconv(data)

        # data = F.softmax(data, dim=1) # dim=1 refers to the 13 microphones

        return data
    
    # def train(self,
    #           dataset: torch.utils.data.Dataset,
    #           lr: float = 0.001,
    #           momentum: float = 0.9,
    #           epochs: int = 100,
    #           early_stopping: int = 3,
    #           mininbatch_size: int = 64):
        
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=mininbatch_size, shuffle=True, num_workers=2)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr, momentum=momentum)
    #     for epoch in range(epochs):
    #         for i, minibatch in tqdm(enumerate(dataloader)):
    #             raw_audio, directions = minibatch

    #             # Forward + backward + optimize
    #             optimizer.zero_grad()
    #             outputs = self.forward(raw_audio)
    #             loss = criterion(outputs, directions)
    #             loss.backward()
    #             optimizer.step()

    #             # Statistics
    #             running_loss += loss.item()
    #             if i % 2000 == 1999:    # print every 2000 mini-batches
    #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #                 running_loss = 0.0

