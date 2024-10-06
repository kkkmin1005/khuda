import torch.nn as nn

'''

Class VGG19
    1. def __init__(self, base_dim = 64):
        1) Hyper parameter
        - base_dim is derived from "VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION."
        - The architecture of this class is partially based on VGG19, with the notable exception that it does not include fully connected layers.
    
    2. def forward(self, x):
        - The forward pass of the VGG19 model.
        - EXAMPLE ) input : 224 x 224 x 3 -> output :  7 x 7 x 512
        
'''


def conv_2(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

# 2개의 conv2d 레이어와 배치 정규화 및 relu 활성화함수를 조합해 합성곱 신경망을 정의함
# 패딩을 통해 출력 이미지의 크기가 입력과 같도록 합니다.
# 맥스 풀링 레이어를 사용하여 특성 맵의 크기를 절반으로 줄입니다.

def conv_4(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

# 위와 비슷하지만 4개의 레이어를 사용해 만든 합성곱 신경망입니다.
# vgg19의 후반부에서 더 깊은 네트워크를 구성하기 위해 정의되었습니다.

class VGG19(nn.Module):
    def __init__(self, base_dim=64):
        super(VGG19, self).__init__()
        self.feature = nn.Sequential(
        conv_2(3, base_dim),
        conv_2(base_dim, base_dim*2),
        conv_4(base_dim*2, base_dim*4),
        conv_4(base_dim*4, base_dim*8),
        conv_4(base_dim*8, base_dim*8)
        )
        
    def forward(self, x):
        x = self.feature(x)
        
        return x
    
# 기본 채널의 수를 결정한 후 위에서 정의한 신경망을 쌓아 특징을 추출합니다.
# FC 레이어가 포함되지 않은 구조로 특징만 추출합니다.