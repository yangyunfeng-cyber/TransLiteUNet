待文章刊出后放出模型细节！
...
...
class TransLiteUNet_V2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(TransLiteUNet_V2, self).__init__()
        features = [32,64,128,256]
        # 改变通道数，尺寸不变。输入数据的维度是[BatchSize, 4, 128,128,128]，输出是[BatchSize, 32, 128,128,128]
        self.inc = InConv(in_channels, features[0])
        # 通道翻倍，尺寸减半。输入[BatchSize, 32, 128,128,128]，
        # 输出[BatchSize, 64, 64,64,64]，其中W= (128-2+2*0)/2+1=64, H=64, D=(128-2+2*0)/2+1=64
        self.down1 = Down(features[0], features[1])
        # 通道翻倍，尺寸减半。输入[BatchSize, 64, 64,64,64]，输出为[BatchSize, 128, 32,32,32]
        self.down2 = Down(features[1], features[2])
        # 通道翻倍，尺寸减半。输入[BatchSize, 128, 32,32,32]，输出为[BatchSize, 256, 16,16,16]
        self.down3 = Down(features[2], features[3])
        # 通道不变，尺寸减半。输入[BatchSize, 256, 16,16,16]，输出为[BatchSize, 256, 8,8,8]
        self.down4 = Down(features[3], features[3])

        # 此处如果需要再优化可以考虑加入位置编码，提高transformer嵌入的维度
        self.LiteTrans = MobileViTBlock(
            in_channels=256,
            transformer_dim=64,
            ffn_dim=128,
            n_transformer_blocks=1,  # 堆叠transformer块的个数
            patch_h=2,
            patch_w=2,
            patch_d=2,
            dropout=0.1,
            ffn_dropout=0.0,
            attn_dropout=0.1,
            head_dim=16,
            conv_ksize=3
        )

        # 输入为[BatchSize, 256, 10,10,8]，输出为[BatchSize, 128, 20,20,16] ,
        # H = (H-1) ×stride−2×padding+kernel_size = (10-1)*2-2*0+2= 20, W=20, D= 16
        self.up1 = Up(features[3], features[3], features[2])
        # 输入为[BatchSize, 128, 20,20,16]，输出为[BatchSize, 64, 40,40,32]
        self.up2 = Up(features[2], features[2], features[1])
        # 输入为[BatchSize, 64, 40,40,32]，输出为[BatchSize, 32, 80,80,64]
        self.up3 = Up(features[1], features[1], features[0])
        # 输入为[BatchSize, 32, 80,80,64]，输出为[BatchSize, 32, 160,160,128]
        self.up4 = Up(features[0], features[0], features[0])
        # 输入为[BatchSize, 32, 160,160,128]，输出为[BatchSize, 4, 160,160,128]
        self.outc = OutConv(features[0], num_classes)
        self.apply(self.init_parameters)

        # 下面这一大段代码都是为了初始化模型参数

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv3d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 输入和输出大小不变
        x6 = self.LiteTrans(x5)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
...
...



