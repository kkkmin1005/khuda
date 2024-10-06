import torch
import torch.nn as nn
import torch.nn.functional as F

'''

Class ArcFace
    1. def __init__(self, in_dim, out_dim, s, m):
        - s and m are parameters derived from "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
        - Matrix W:
            1) The matrix W has dimensions in_dim x out_dim.
            2) W is initialized using Xavier initialization.
            3) in_dim: Dimensionality of the tensor resulting from flattening the forward pass of VGG19.
            4) out_dim: Number of classes.
            
    2. def forward(self, x):
        - the forward pass of the ArcFace model.

'''

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m):
        super(ArcFace, self).__init__()
        self.s = s # 스케일링 파라미터
        self.m = m # 각도 마진
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        nn.init.kaiming_uniform_(self.W)
        
    def forward(self, x):
        normalized_x = F.normalize(x, p=2, dim=1) # 입력 특징 벡터 x를 받은 후 정규화를 진행합니다. 
        normalized_W = F.normalize(self.W, p=2, dim=0)
    
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        
        # Using torch.clamp() to ensure cosine values are within a safe range,
        # preventing potential NaN losses.
        
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        probability = self.s * torch.cos(theta+self.m)
        
        return probability
    
# 입력 벡터 x와 클래스별 가중치 행렬 w사이의 코사인 유사도를 기반으로 각도를 계산하고
# 이 각도에 마진을 더하여 최종적으로 더 큰 분리도를 가진 모델을 학습합니다.