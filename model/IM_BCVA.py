import torch
from einops import repeat, rearrange
from torch import nn
from model.OctModel import OctNet
from model.SloModel import SloNet

class IncompleteBCVA(nn.Module):
    def __init__(self, cfgs):
        super(IncompleteBCVA, self).__init__()
        dim = cfgs['model_cfg']['complete_fusion']['dim']
        classes = cfgs['model_cfg']['BCVA_Num_Classes']
        self.cfgs = cfgs
        self.isReWeighting = cfgs['base_cfg']['isReWeighting']
        self.octEncoder = OctNet(cfgs)
        if self.isReWeighting:
            self.octReWeighting = OctReWeighting()
        self.sloEncoder = SloNet(cfgs)
        self.fusionNetwork  = ModalFusion(cfgs['base_cfg']['isAttentionMask'], **cfgs['model_cfg']['incomplete_fusion'])
        self.pred = nn.Sequential(nn.LayerNorm(dim),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(dim, classes),
                                  )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self,
                OctImage, # [batchSize, channel, h, w]
                OTSU, # [batchSize, channel, h, w]
                patientMessage, # [batchSize, seqLength, dim]
                SloImage, # [batchSize, channel, h, w]
                MissingLabel, # [batchSize, 3]
                diagOct,
                diagSlo,
                ROI = None,
                ):
        octEmbed, predOct = self.octEncoder(OctImage, OTSU, ROI)
        sloEmbed, predSlo = self.sloEncoder(SloImage)
        textEmbed = patientMessage
        if self.isReWeighting:
            octEmbed = self.octReWeighting(octEmbed, diagOct)

        fusionFeatures = self.fusionNetwork([textEmbed, octEmbed, sloEmbed], MissingLabel)


        valid_index_mask = torch.cat([torch.ones(octEmbed.size(0), size).cuda() * label.unsqueeze(1)
                                      for size, label in zip(self.cfgs['model_cfg']['incomplete_fusion']['seqLs'], MissingLabel.T)], dim = 1)


        valid_features = fusionFeatures * valid_index_mask.unsqueeze(-1)

        valid_feature_count = valid_index_mask.sum(dim = 1, keepdim = True)
        valid_feature_count[valid_feature_count == 0] = 1
        mean_valid_features = valid_features.sum(dim = 1) / valid_feature_count
        predBCVA = self.pred(mean_valid_features).squeeze()
        return predBCVA, predOct, predSlo, octEmbed, sloEmbed
class MaskAttention(nn.Module):
    def __init__(self, isAttentionMask, dim, heads = 8, dim_head = 64, dropout = 0., seqLs = []):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.isAttentionMask = isAttentionMask
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = SoftMax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.seqLs = seqLs

    def get_mask(self, modality_labels):
        batch_size, num_modalities = modality_labels.shape  # [B, num_modality]
        device = modality_labels.device
        total_length = sum(self.seqLs)  # [1, 196, 196, 196]
        # Initialize the mask with zeros (will be filled with -inf for missing modalities)
        mask = torch.zeros((batch_size, total_length, total_length), device = device)

        # Calculate the start and end indices for each modality
        end_indices = torch.cumsum(torch.tensor(self.seqLs, device = device), dim = 0)
        start_indices = torch.cat((torch.tensor([0], device = device), end_indices[:-1]))

        # Iterate over each modality and update the mask
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            # Check where the modality is missing (0 in modality_labels)
            modality_missing = modality_labels[:, i].unsqueeze(1).unsqueeze(2) == 0  # [B] -> [B, 1, 1]

            # Update the mask for missing modalities
            # Expand the mask to cover the entire rows and columns for the missing modality

            mask[:, start_idx:end_idx, :] = mask[:, start_idx:end_idx, :].masked_fill(modality_missing, float("-inf"))  # row
            mask[:, :, start_idx:end_idx] = mask[:, :, start_idx:end_idx].masked_fill(modality_missing, float("-inf"))  # column

        return mask

    def forward(self, x, m_labels):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.isAttentionMask:
            mask = self.get_mask(m_labels).unsqueeze(1)  # [B, 1, N1, N2]
            mask = repeat(mask, 'b 1 n1 n2 -> b h n1 n2', h = dots.shape[1])  # broadcast to h heads
            dots = dots + mask
        attn = self.attend(dots)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class FusionTransformer(nn.Module):
    def __init__(self, isAttentionMask, dim, depth, heads, dim_head, mlp_dim, dropout = 0., seqLs = []):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MaskAttention(isAttentionMask, dim, heads = heads, dim_head = dim_head, dropout = dropout, seqLs = seqLs),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, m_labels):
        for attn, ff in self.layers:
            x = attn(x, m_labels) + x

            x = ff(x) + x

        return self.norm(x)

class ModalFusion(nn.Module):
    def __init__(self, isAttentionMask, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., seqLs = []):
        super(ModalFusion, self).__init__()
        self.modality_emb = nn.ParameterList([nn.Parameter(torch.randn(1, 1, dim)) for _ in range(len(seqLs))])
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FusionTransformer(isAttentionMask, dim, depth, heads, dim_head, mlp_dim, dropout, seqLs)
        total_seqL = sum(seqLs)
        self.pos_embedding = nn.Parameter(torch.randn(1, total_seqL, dim))

    def forward(self, feature_list, m_labels):
        # add modality embeddings
        for i in range(len(feature_list)):
            b, n, _ = feature_list[i].shape  # [B, N, C]
            modality_emb = repeat(self.modality_emb[i], '1 1 d -> b n d', b = b, n = n)
            feature_list[i] = feature_list[i] + modality_emb
        features = torch.cat(feature_list, dim = 1)  # [B, total_seqL, C]
        # add positional embeddings, optional
        features = features + self.pos_embedding
        features = self.dropout(features)

        features = self.transformer(features, m_labels)  # [B, total_seqL, C]

        return features

class SoftMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, inputData):
        all_neg_inf_rows = torch.all(torch.isinf(inputData), dim = self.dim)
        inputData[all_neg_inf_rows] = 0
        max_item, _ = torch.max(inputData, dim = self.dim, keepdim = True)
        data = inputData - max_item
        x_exp = torch.exp(data)
        result = x_exp / torch.sum(x_exp, dim = self.dim, keepdim = True)
        result[all_neg_inf_rows] = 0
        return result

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class OctReWeighting(nn.Module):
    def __init__(self, dim = 768, heads = 6, dim_head = 128, dropout = 0.):
        super(OctReWeighting, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_key = nn.Linear(dim, inner_dim, bias = False)
        self.to_query = nn.Linear(dim, inner_dim, bias = False)
    def forward(self, Oct, diagOct):
        b, n, c = Oct.shape
        key = self.to_key(self.norm(Oct))
        query = self.to_query(diagOct)
        dots = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        weights = torch.mean(attn, dim = 1)
        weights = repeat(weights.unsqueeze(-1), 'b n 1 -> b n c', b = b, n = n, c = c)
        OctReWeightedEmbed = Oct * weights
        return OctReWeightedEmbed

