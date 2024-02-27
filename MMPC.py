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