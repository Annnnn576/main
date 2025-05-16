# from .nn_utils import *
# from utils_algo import *
from preact_resnet import *
from compact_bilinear_pooling import *
from torchvision.models import resnet50


class Model(nn.Module):
    def __init__(self, head='linear', feat_dim=768):
        super().__init__()
        dim_in = 1000
        feature_extractor = resnet50(pretrained=True)
        # for param in feature_extractor.parameters():
        #     param.requires_grad = False
        feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.encoder = feature_extractor

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.decoder = torch.nn.Linear(in_features=1000, out_features=20)

    def forward(self, x):
        feat = self.encoder(x)
        z = F.normalize(self.head(feat), dim=1)
        # feat = self.head(feat)
        logits = self.decoder(feat)
        # result = torch.cat((feat, z), dim=1)
        return z, logits #



class SemanticDecouple(nn.Module):
    """
    Semantic-Special Feature
    """
    def __init__(self,
                 num_classes,
                 feature_dim,
                 semantic_dim,
                 intermediary_dim=1024):
        super(SemanticDecouple, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.intermediary_dim = intermediary_dim

        self.feature_trans = nn.Linear(self.feature_dim,
                                       self.intermediary_dim,
                                       bias=False)
        self.semantic_trans = nn.Linear(self.semantic_dim,
                                        self.intermediary_dim,
                                        bias=False)
        self.joint_trans = nn.Linear(self.intermediary_dim,
                                     self.intermediary_dim)

    def forward(self, global_feature, semantic_feature):
        """
        :param global_feature:  N*d
        :param semantic_feature:  C*k
        :return: N*C*d'
        """
        (n, d) = global_feature.shape
        (c, k) = semantic_feature.shape
        global_trans_feature = self.feature_trans(global_feature)
        semantic_trans_feature = self.semantic_trans(semantic_feature)
        global_trans_feature = global_trans_feature.unsqueeze(0).repeat(
            c, 1, 1).transpose(0, 1)
        semantic_trans_feature = semantic_trans_feature.unsqueeze(0).repeat(
            n, 1, 1)
        joint_trans_feature = torch.mul(
            global_trans_feature,
            semantic_trans_feature).contiguous().view(n * c, -1)
        semantic_special_feature = self.joint_trans(
            torch.tanh(joint_trans_feature)).contiguous().view(n, c, -1)
        return semantic_special_feature

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output