import torch
import torch.nn as nn
from model.network.rn import RN_B,RN_L
# from .network.rn import RN_B, RN_L



#实现残差块
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class saRN_ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, threshold, use_spectral_norm=True):
        super(saRN_ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            RN_L(feature_channels=1024, threshold=threshold),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=1024, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

# 实现double conv
class Double_conv_down(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''

    def __init__(self, in_ch, out_ch):
        super(Double_conv_down, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            RN_B(feature_channels=out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            RN_B(feature_channels=out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class Double_conv_up(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''

    def __init__(self, in_ch, out_ch):
        super(Double_conv_up, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            RN_L(feature_channels=out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            RN_L(feature_channels=out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# 实现input conv
class In_conv(nn.Module):
    ''' input conv layer
        let input 3 channels image to 64 channels
        The oly difference between `inconv` and `down` is maxpool layer
    '''
    def __init__(self, in_ch, out_ch):
        super(In_conv, self).__init__()
        self.conv = Double_conv_down(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    ''' normal down path
        MaxPool2d => double_conv
    '''
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv_down(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(Up, self).__init__()

        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=1),
                                    nn.LeakyReLU(inplace=True))
        self.conv = Double_conv_up(in_ch, out_ch)

    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.inc = In_conv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # # #middle
        # blocks = []
        # for _ in range(residual_blocks):
        #     # block = ResnetBlock(256, 2, use_spectral_norm=False)
        #     block = saRN_ResnetBlock(1024, dilation=2, threshold=threshold, use_spectral_norm=False)
        #     blocks.append(block)
        #
        # self.middle = nn.Sequential(*blocks)



        self.up1 = Up(1024, 512, Transpose=True)
        self.up2 = Up(512, 256, Transpose=True)
        self.up3 = Up(256, 128, Transpose=True)
        self.up4 = Up(128, 64, Transpose=True)

        self.outc = OutConv(64, 1)

    def forward(self, x0):
        # 计算掩码
        mask = self.get_mask(x0)
        x1 = self.inc(x0)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # x6 = self.middle(x5)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        # 应用掩码
        y = x0 + mask * x
        return y

    def get_mask(self, x):
        sam_add = x.sum(dim=2) == 0
        mask_sam = torch.where(sam_add, 1, 0)
        mask_sam = mask_sam.unsqueeze(2)
        return mask_sam


if __name__ == '__main__':
    net = Unet()
    print(net)