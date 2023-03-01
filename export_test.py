import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
from torch.onnx.symbolic_helper import parse_args 
import torch.onnx

class PointCloudSample(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0]
    
    @staticmethod
    def symbolic(
        g, 
        input_xyzs, 
        input_xyz_num, 
        sample_xyz_num,
        sample_method,
    ):
        # return sample_index
        return g.op(
            "point_cloud_sample", 
            input_xyzs_i = 0, 
            input_xyz_num_i = 0,#input_xyz_num, 
            sample_xyz_num_i = 0,#sample_xyz_num,
            sample_method_i = 0,#sample_method,      
        )

pcs = PointCloudSample.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(x)
        x = x.view(-1)
        x = pcs(x, 1, 1, 0)
        return x

net = TinyNet().cuda()
ipt = torch.ones(2,3,12,12).cuda()
torch.onnx.export(net, (ipt,), 'tinynet.onnx', opset_version=11, enable_onnx_checker=False)
print(onnx.load('tinynet.onnx'))
