from Backbone import VGG19
from ArcFace import ArcFace
import torch.nn as nn

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.ArcFace = ArcFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        x = self.ArcFace(x)
        
        return x
    
# vgg19에서 이미지의 특징을 추출합니다. 특징 맵은 7*7*512의 크기로 출력되며, 이를 1차원 데이터로 펼칩니다.
# 특징이 arcface레이어로 전달되어 클래스에 대한 확률을 계산합니다.