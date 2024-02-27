class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class MMPC(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        h_x, w_y, x = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        hw = torch.cat([h_x, w_y], dim=2)
        hw = self.conv_hw(hw)
        h_x, w_y = torch.split(hw, [h, w], dim=2)
        hw_weight = self.conv_pool_hw(hw).sigmoid()
        path = x * torch.mean(hw_weight, dim=2, keepdim=True)
        h_x_weight, w_y_weight = torch.split(hw_weight, [h, w], dim=2)
        h_x, w_y = h_x * h_x_weight * path, w_y * w_y_weight * path
        # h_x, w_y = h_x * h_x_weight, w_y * w_y_weight * path         # turn off path1
        # h_x, w_y = h_x * h_x_weight * path, w_y * w_y_weight         # turn off path2
        # return x * h_x.sigmoid() * w_y.permute(0, 1, 3, 2).sigmoid() # turn off path3
        return x * h_x.sigmoid() * w_y.permute(0, 1, 3, 2).sigmoid() * path.sigmoid()
