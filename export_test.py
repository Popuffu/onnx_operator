import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
from torch.onnx.symbolic_helper import parse_args 
import torch.onnx

class PCSample(Function):
    @staticmethod
    def forward(ctx, input_xyzs, 
                 input_xyz_num, sample_xyz_num, sample_method):
        return input_xyzs + 1
    
    @staticmethod
    def symbolic(g, input_xyzs, 
                 input_xyz_num, sample_xyz_num, sample_method):
        # return sample_index
        return g.op(
            "pc_sample",
            input_xyzs,
            input_xyz_num_i = input_xyz_num,
            sample_xyz_num_i = sample_xyz_num,
            sample_method_s = sample_method,
        )
pcsample_op = PCSample.apply
class PCSampleLayer(nn.Module):
   def __init__(self, input_xyz_num, sample_xyz_num, sample_method):
        super(PCSampleLayer, self).__init__()
        self.input_xyz_num = input_xyz_num,
        self.sample_xyz_num = sample_xyz_num,
        self.sample_method = sample_method,

   def forward(self, input_xyzs):
       return pcsample_op(input_xyzs, self.input_xyz_num, self.sample_xyz_num, self.sample_method)
   

class PCQuery(Function):
    @staticmethod
    def forward(ctx, input_xyzs, query_xyz_index, 
                 input_xyz_num, query_xyz_num, query_method, query_num, radius):
        return input_xyzs + query_xyz_index
    
    @staticmethod
    def symbolic(g, input_xyzs, query_xyz_index, 
                 input_xyz_num, query_xyz_num, query_method, query_num, radius):
        # return group_index
        return g.op(
            "pc_query",
            input_xyzs,
            query_xyz_index,
            input_xyz_num_i = input_xyz_num,
            query_xyz_num_i = query_xyz_num,
            query_method_s = query_method,
            query_num_i = query_num,
            radius_f = radius,
        )
pcquery_op = PCQuery.apply
class PCQueryLayer(nn.Module):
   def __init__(self, input_xyz_num, query_xyz_num, query_method, query_num, radius):
        super(PCQueryLayer, self).__init__()
        self.input_xyz_num = input_xyz_num,
        self.query_xyz_num = query_xyz_num,
        self.query_method = query_method,
        self.query_num = query_num,
        self.radius = radius

   def forward(self, input_xyzs, query_xyz_index):
       return pcquery_op(input_xyzs, query_xyz_index, self.input_xyz_num, self.query_xyz_num, self.query_method, self.query_num, self.radius)


class PCFetch(Function):
    @staticmethod
    def forward(ctx, input_data, fetch_index):
        return input_data + fetch_index
    
    @staticmethod
    def symbolic(g, input_data, fetch_index):
        # return output_data
        return g.op(
            "pc_fetch",
            input_data,
            fetch_index
        )
pcfetch_op = PCFetch.apply
class PCFetchLayer(nn.Module):
   def __init__(self):
        super(PCFetchLayer, self).__init__()

   def forward(self, input_data, fetch_index):
       return pcfetch_op(input_data, fetch_index)


class PCMax(Function):
    @staticmethod
    def forward(ctx, input_data, dim):
        return input_data + 1
    
    @staticmethod
    def symbolic(g, input_data, dim):
        # return output_data
        return g.op(
            "pc_max",
            input_data,
            dim_i = dim,
        )
pcmax_op = PCMax.apply
class PCMaxLayer(nn.Module):
   def __init__(self):
        super(PCMaxLayer, self).__init__()

   def forward(self, input_data, dim):
       return pcmax_op(input_data, dim)


class PCConcat(Function):
    @staticmethod
    def forward(ctx, input_a, input_b):
        return input_a + input_b
    
    @staticmethod
    def symbolic(g, input_a, input_b):
        # return output_feature
        return g.op(
            "pc_concat",
            input_a,
            input_b,
        )
pcconcat_op = PCConcat.apply
class PCConcatLayer(nn.Module):
   def __init__(self):
        super(PCConcatLayer, self).__init__()

   def forward(self, input_a, input_b):
       return pcconcat_op(input_a, input_b)


class PCConv2d(Function):
    @staticmethod
    def forward(ctx, input_feature, 
                weight_dir, bias_dir, height, width, cin, cout, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bias_flag, bn_flag, relu_flag):
        return input_feature + 1
    
    @staticmethod
    def symbolic(g, input_feature, 
                 weight_dir, bias_dir, height, width, cin, cout, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bias_flag, bn_flag, relu_flag):
        # return output_feature
        return g.op(
            "pc_denseconv2d",
            input_feature,
            weight_s = weight_dir,
            bias_s = bias_dir,
            height_i = height,
            width_i = width,
            cin_i = cin,
            cout_i = cout,
            kernel_h_i = kernel_h,
            kernel_w_i = kernel_w,
            stride_h_i = stride_h,
            stride_w_i = stride_w,
            pad_h_i = pad_h,
            pad_w_i = pad_w,
            bias_flag_i = bias_flag,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
        )
pcconv2d_op = PCConv2d.apply
class PCConv2dLayer(nn.Module):
   def __init__(self, weight_dir, bias_dir, height, width, cin, cout, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bias_flag, bn_flag, relu_flag):
        super(PCConv2dLayer, self).__init__()
        self.weight_dir = weight_dir,
        self.bias_dir = bias_dir,
        self.height = height,
        self.width = width,
        self.cin = cin,
        self.cout = cout,
        self.kernel_h = kernel_h,
        self.kernel_w = kernel_w,
        self.stride_h = stride_h,
        self.stride_w = stride_w,
        self.pad_h = pad_h,
        self.pad_w = pad_w,
        self.bias_flag = bias_flag,
        self.bn_flag = bn_flag,
        self.relu_flag = relu_flag

   def forward(self, input_feature):
       return pcconv2d_op(input_feature, self.weight_dir, self.bias_dir, self.height, self.width, self.cin, 
                          self.cout, self.kernel_h, self.kernel_w, self.stride_h, self.stride_w, self.pad_h, 
                          self.pad_w, self.bias_flag, self.bn_flag, self.relu_flag)


class PCKernelMapping(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0] + 1
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        kernel,
        stride,
        pad,
        indice_key,
        submconv_flag,

    ):
        # return mapping_table
        return g.op(
            "pc_kernel_mapping",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            kernel_i = kernel,
            stride_i = stride,
            pad_i = pad,
            indice_key_i = indice_key,
            submconv_flag_i = submconv_flag,
        )


class PCSubmConv3d(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0] + 1
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        voxel_features,
        cin,
        cout,
        kernel,
        mapping_table,
        weight,
        bias_flag,
        bias,
        bn_flag,
        relu_flag,

    ):
        # return output_voxel_num, output_voxel_xyz, output_voxel_feature
        return g.op(
            "pc_submconv3d",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            voxel_features_i = voxel_features,
            cin_i = cin,
            cout_i = cout,
            kernel_i = kernel,
            mapping_table_i = mapping_table,
            weight_i = weight,
            bias_flag_i = bias_flag,
            bias_i = bias,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
        )


class PCSparseConv3d(Function):
    @staticmethod
    def forward(ctx, *inputs):
        return inputs[0] + 1
    
    @staticmethod
    def symbolic(
        g, 
        voxel_num,
        voxel_xyzs,
        voxel_features,
        cin,
        cout,
        kernel,
        mapping_table,
        weight,
        bias_flag,
        bias,
        bn_flag,
        relu_flag,

    ):
        # return output_voxel_num, output_voxel_xyz, output_voxel_feature
        return g.op(
            "pc_sparseconv3d",
            voxel_num_i = voxel_num,
            voxel_xyzs_i = voxel_xyzs,
            voxel_features_i = voxel_features,
            cin_i = cin,
            cout_i = cout,
            kernel_i = kernel,
            mapping_table_i = mapping_table,
            weight_i = weight,
            bias_flag_i = bias_flag,
            bias_i = bias,
            bn_flag_i = bn_flag,
            relu_flag_i = relu_flag,
        )





class PointNetPlusPlus(nn.Module):
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.fps0 = PCSampleLayer(input_xyz_num=4096, sample_xyz_num=1024, sample_method="fps")
        self.fps1 = PCSampleLayer(input_xyz_num=1024, sample_xyz_num=512, sample_method="fps")
        self.fps2 = PCSampleLayer(input_xyz_num=512, sample_xyz_num=128, sample_method="fps")
        
        self.fetch = PCFetchLayer()

        self.ball_query0 = PCQueryLayer(input_xyz_num=1024, query_xyz_num=512, query_method="ball_query", query_num=32, radius=0.2)
        self.ball_query1 = PCQueryLayer(input_xyz_num=512, query_xyz_num=128, query_method="ball_query", query_num=64, radius=0.4)

        self.concat = PCConcatLayer()

        self.max = PCMaxLayer()

        self.conv0_0 = PCConv2dLayer(weight_dir="conv0_0_weight_path", bias_dir="conv0_0_bias_path", 
                                      height=512, width=32, cin=3, cout=64, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv0_1 = PCConv2dLayer(weight_dir="conv0_1_weight_path", bias_dir="conv0_1_bias_path", 
                                      height=512, width=32, cin=64, cout=64, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv0_2 = PCConv2dLayer(weight_dir="conv0_2_weight_path", bias_dir="conv0_2_bias_path", 
                                      height=512, width=32, cin=64, cout=128, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        
        self.conv1_0 = PCConv2dLayer(weight_dir="conv1_0_weight_path", bias_dir="conv1_0_bias_path", 
                                      height=128, width=64, cin=131, cout=128, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv1_1 = PCConv2dLayer(weight_dir="conv1_1_weight_path", bias_dir="conv1_1_bias_path", 
                                      height=128, width=64, cin=128, cout=128, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv1_2 = PCConv2dLayer(weight_dir="conv1_2_weight_path", bias_dir="conv1_2_bias_path", 
                                      height=128, width=64, cin=128, cout=256, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        
        self.conv2_0 = PCConv2dLayer(weight_dir="conv2_0_weight_path", bias_dir="conv2_0_bias_path", 
                                      height=128, width=1, cin=259, cout=256, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv2_1 = PCConv2dLayer(weight_dir="conv2_1_weight_path", bias_dir="conv2_1_bias_path", 
                                      height=128, width=1, cin=256, cout=512, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.conv2_2 = PCConv2dLayer(weight_dir="conv2_2_weight_path", bias_dir="conv2_2_bias_path", 
                                      height=128, width=1, cin=512, cout=1024, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        
        self.fc0 = PCConv2dLayer(weight_dir="fc0_weight_path", bias_dir="fc0_bias_path", 
                                      height=1, width=1, cin=1024, cout=512, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.fc1 = PCConv2dLayer(weight_dir="fc1_weight_path", bias_dir="fc1_bias_path", 
                                      height=1, width=1, cin=512, cout=256, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        self.fc2 = PCConv2dLayer(weight_dir="fc2_weight_path", bias_dir="fc2_bias_path", 
                                      height=1, width=1, cin=256, cout=40, kernel_h=1, kernel_w=1, stride_h=1, 
                                      stride_w=1, pad_h=0, pad_w=0, bias_flag=1, bn_flag=1, relu_flag=1)
        
    def forward(self, xyzs):
        index_after_fps0 = self.fps0(xyzs)
        xyzs_after_fps0 = self.fetch(xyzs, index_after_fps0)

        index_after_fps1 = self.fps1(xyzs_after_fps0)
        xyzs_after_fps1 = self.fetch(xyzs_after_fps0, index_after_fps1)

        index_after_fps2 = self.fps2(xyzs_after_fps1)
        xyzs_after_fps2 = self.fetch(xyzs_after_fps1, index_after_fps2)

        index_after_ballquery0 = self.ball_query0(xyzs_after_fps0, index_after_fps1)
        xyzs_after_ballquery0 = self.fetch(xyzs_after_fps0, index_after_ballquery0)

        index_after_ballquery1 = self.ball_query1(xyzs_after_fps1, index_after_fps2)
        xyzs_after_ballquery1 = self.fetch(xyzs_after_fps1, index_after_ballquery1)

        feat_group_0 = self.conv0_0(xyzs_after_ballquery0)
        feat_group_0 = self.conv0_1(feat_group_0)
        feat_group_0 = self.conv0_2(feat_group_0)
        feat_group_0 = self.max(feat_group_0, dim=1)

        feat_group_1 = self.concat(feat_group_0, xyzs_after_ballquery1)
        feat_group_1 = self.conv1_0(feat_group_1)
        feat_group_1 = self.conv1_1(feat_group_1)
        feat_group_1 = self.conv1_2(feat_group_1)
        feat_group_1 = self.max(feat_group_1, dim=1)

        feat_group_2 = self.concat(feat_group_1, xyzs_after_fps2)
        feat_group_2 = self.conv2_0(feat_group_2)
        feat_group_2 = self.conv2_1(feat_group_2)
        feat_group_2 = self.conv2_2(feat_group_2)
        feat_group_2 = self.max(feat_group_2, dim=0)
        
        feat_fc = self.fc0(feat_group_2)
        feat_fc = self.fc1(feat_fc)
        feat_fc = self.fc2(feat_fc)
        return feat_fc
    

net = PointNetPlusPlus()
points = torch.ones(4096,3)
export_name = "pointnetplusplus.onnx"
torch.onnx.export(net, (points,), export_name, opset_version=11, input_names=["input_xyzs",], output_names=["OutputTensor"], enable_onnx_checker=False)
# torch.onnx.export(net, (points,), 'tinynet.onnx', opset_version=13, input_names=["xyzs",], output_names=["OutputTensor",], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
print("Finish export to %s" % export_name)

