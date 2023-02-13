import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PCCT"]


class OA(nn.Module):
    """Offset-Attention Module (OA)"""

    def __init__(self, d: int):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(d, d // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(d, d // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        self.v_conv = nn.Conv1d(d, d, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.tconv = nn.Conv1d(d, d, 1)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x: torch.Tensor):
        # x (B, D, N)
        # (B, D//4, N).permute(0, 2, 1) -> (B, N, D//4)
        x_q = self.q_conv(x).permute(0, 2, 1)
        # B, D//4, N
        x_k = self.k_conv(x)
        # B, D, N
        x_v = self.v_conv(x)
        # B, N, N
        energy = torch.bmm(x_q, x_k)
        # B, N, N
        attention = self.softmax(energy)
        # B, N, N
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        # B, D, N
        x_r = torch.bmm(x_v, attention)
        # B, D, N
        x_r = F.relu(self.bn(self.tconv(x - x_r)))
        # B, D, N
        x = x + x_r
        # B, D, N
        return x


class BasicCA(nn.Module):
    """Basic Cascade-Attention Module (CA)"""

    def __init__(self, d: int):
        super(BasicCA, self).__init__()

        self.oa = OA(d)

    def forward(self, x: torch.Tensor):
        # x (B, d, N)
        # B, d, N
        at = F.relu(self.oa(x))
        # B, d, N
        return at


class CA(nn.Module):
    """Cascade-Attention Module (CA)"""

    def __init__(self, d: int, blocks: int):
        super(CA, self).__init__()

        # Cascade-Attention
        self.layers = nn.ModuleList()
        for _ in range(blocks):
            self.layers.append(BasicCA(d))

    def forward(self, x: torch.Tensor):
        # x (B, d, N)
        # [(B, d, N),]
        ats = [self.layers[0](x)]
        for layer in self.layers[1:]:
            # .append[(B, d, N)]
            ats.append(layer(ats[-1]))
        # B, d+d+d+d+..., N
        x = torch.cat(ats, dim=1)
        # B, d+d+d+d+..., N
        return x


class STNCA(nn.Module):
    """STN with Cascade-Attention module (STNCA)"""

    def __init__(self, d: int, with_ca: bool = False, blocks: int = 4):
        super(STNCA, self).__init__()

        self.d: int = d
        self.with_ca: bool = with_ca

        self.conv1 = nn.Conv1d(d, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        if with_ca:
            # Cascade-Attention
            self.ca = CA(128, blocks=blocks)

            self.conv3 = nn.Conv1d(512, 1024, 1)
        else:
            self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, d * d)

    def forward(self, x: torch.Tensor):
        # x (B, 3, N)
        B = x.size()[0]
        # B, 64, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, 128, N
        x = F.relu(self.bn2(self.conv2(x)))
        if self.with_ca:
            # B, 512, N
            x = self.ca(x)
        # B, 1024, N
        x = F.relu(self.bn3(self.conv3(x)))
        # B, 1024, 1
        x = torch.max(x, 2, keepdim=True)[0]
        # B, 1024
        x = x.view(-1, 1024)
        # B, 512
        x = F.relu(self.bn4(self.fc1(x)))
        # B, 256
        x = F.relu(self.bn5(self.fc2(x)))
        # B, d x d
        x = self.fc3(x)
        # B, d x d
        iden = torch.eye(self.d, dtype=torch.float32,
                         device=x.device).view(1, self.d * self.d).repeat(B, 1)
        # B, d x d
        x = x + iden
        # B, d, d
        x = x.view(-1, self.d, self.d)
        # B, d, d
        return x


class PCCTfeat(nn.Module):
    """Point cloud classification based on transformer modeule (PCCT)"""

    def __init__(self, feature_transform: bool = False, blocks: int = 4):
        super(PCCTfeat, self).__init__()

        self.feature_transform: bool = feature_transform

        self.stn3d = STNCA(d=3, with_ca=False)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.oa = OA(64)

        if feature_transform:
            self.stn64d = STNCA(d=64, with_ca=True, blocks=blocks)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        # Cascade-Attention
        self.ca = CA(128, blocks=blocks)

        self.conv3 = torch.nn.Conv1d(blocks * 128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor, save_critical_points: bool = False):
        # x ( B, 3, N)
        B, D, N = x.size()

        # Originl points without spatial transformation
        # B, N, 3
        orig_x = x.transpose(2, 1)
        # B, 3, 3
        trans = self.stn3d(x)
        # B, N, 3
        x = torch.bmm(orig_x, trans)
        # B, 3, N
        x = x.transpose(2, 1)
        # B, 64, N
        x = F.relu(self.bn1(self.conv1(x)))

        # B, 64, N
        at = F.relu(self.oa(x))
        # B, 64, N
        x = x + at

        if self.feature_transform:
            # B, 64, 64
            trans_feat = self.stn64d(x)
            # B, N, 64
            x = x.transpose(2, 1)
            # B, N, 64
            x = torch.bmm(x, trans_feat)
            # B, 64, N
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # B, 128, N
        x = F.relu(self.bn2(self.conv2(x)))

        # B, 512, N
        x = self.ca(x)

        # B, 1024, N
        x = self.bn3(self.conv3(x))
        # B, 1024, 1
        x, index = torch.max(x, 2, keepdim=True)

        if save_critical_points:
            # Save numpy array of the critical points that
            # contributes to max pooling from the last epoch.
            # B, 1024
            index = index.view(-1, 1024)
            index = index.cpu().detach().numpy()

            orig_x = orig_x.cpu().detach().numpy()

            allPoints = []
            critPoints = []
            for i in range(x.shape[0]):
                allPoints.append(orig_x[i])
                critPoints.append(orig_x[i][index[i]])

            # B, N, 3
            allPoints = np.array(allPoints)
            # B, 1024, 3
            critPoints = np.array(critPoints)

            np.savez('all_pts.npz', points=allPoints)
            np.savez('critical_pts.npz', points=critPoints)

        # B, 1024
        x = x.view(-1, 1024)
        # B x 1024, B x 3 x 3, B x 64 x 64 | None
        return x, trans, trans_feat


class PCCT(nn.Module):
    def __init__(self, k: int = 2, p: float = 0.3, feature_transform: bool = False, blocks: int = 4):
        super(PCCT, self).__init__()

        self.feat = PCCTfeat(
            feature_transform=feature_transform, blocks=blocks)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=p)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k)

    def forward(self, x: torch.Tensor, save_critical_points: bool = False):
        # x (B, D, N)

        # B x 1024, B x 3 x 3, B x K=64 x K=64 | None
        x, trans, trans_feat = self.feat(
            x, save_critical_points=save_critical_points)
        # B, 512
        x = F.relu(self.bn1(self.fc1(x)))
        # B, 256
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # B, K
        x = self.fc3(x)
        # B x K
        x = F.log_softmax(x, dim=1)
        # B x K, B x 64 x 64 | None
        return x, trans, trans_feat


def feature_transform_regularizer(x: torch.Tensor):
    # x (B, D, N)
    D = x.size()[1]
    I = torch.eye(D, dtype=torch.float32, device=x.device)[None, :, :]
    loss = torch.mean(torch.norm(
        torch.bmm(x, x.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
