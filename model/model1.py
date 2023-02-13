import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PointNetCls", "feature_transform_regularizer"]


class STN(nn.Module):
    """Spatial Transformer Network module (STN)."""

    def __init__(self, d: int):
        super(STN, self).__init__()

        self.d: int = d

        self.conv1 = nn.Conv1d(d, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, d * d)

    def forward(self, x: torch.Tensor):
        # x (B, 3, N)
        B = x.shape[0]
        # B, 64, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, 128, N
        x = F.relu(self.bn2(self.conv2(x)))
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


class PointNetClsfeat(nn.Module):
    def __init__(self, feature_transform: bool = False):
        super(PointNetClsfeat, self).__init__()

        self.feature_transform: bool = feature_transform

        self.stn3d = STN(3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        if self.feature_transform:
            self.stn64d = STN(64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor, save_critical_points: bool = False):
        # x  (B, 3, N)

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
        x = F.relu(self.bn2(self.conv2(x)))

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

        # B, 64, N
        x = F.relu(self.bn3(self.conv3(x)))
        # B, 128, N
        x = F.relu(self.bn4(self.conv4(x)))
        # B, 1024, N
        x = self.bn5(self.conv5(x))
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


class PointNetCls(nn.Module):
    def __init__(self, k: int = 2, p: float = 0.3, feature_transform: bool = False):
        super(PointNetCls, self).__init__()

        self.feat = PointNetClsfeat(feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=p)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k)

    def forward(self, x: torch.Tensor, save_critical_points: bool = False):
        # B x 1024, B x 3 x 3, B x 64 x 64 | None
        x, trans, trans_feat = self.feat(x, save_critical_points)
        # B, 512
        x = F.relu(self.bn1(self.fc1(x)))
        # B, 256
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        # B, K
        x = self.fc3(x)
        # B x K, B x 3 x 3, B x 64 x 64 | None
        return F.log_softmax(x, dim=1), trans, trans_feat


def feature_transform_regularizer(x: torch.Tensor):
    # x (B, D, N)
    D = x.shape[1]
    I = torch.eye(D, dtype=torch.float32, device=x.device)[None, :, :]
    loss = torch.mean(torch.norm(
        torch.bmm(x, x.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
