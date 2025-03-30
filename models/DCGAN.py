class Generator(nn.Module):
    def __init__(self, 
                 out_channels=3,
                 in_channels_latent_dim=100):
        super().__init__()

        self.out_channels = out_channels
        self.latent_dim = in_channels_latent_dim
        
        self.generator = nn.Sequential(

            ### (B x 100 x 1 x 1) -> (B x 1024 x 4 x 4) ###
            nn.ConvTranspose2d(self.latent_dim, 1024, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            ### (B x 1024 x 4 x 4) -> (B x 512 x 8 x 8)###
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            ### (B x 512 x 8 x 8) -> (B x 256 x 16 x 16) ###
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            ### (B x 256 x 16 x 16) -> (B x 128 x 32 x 32) ###
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            ### (B x 128 x 32 x 32) -> (B x 3 x 64 x 64) ###
            nn.ConvTranspose2d(128, self.out_channels, kernel_size=4, stride=2, padding=1), 
            nn.Tanh()
        )

        self.apply(_init_weights)
        
    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self,
                 in_channels=3):
        super().__init__()

        self.discriminator = nn.Sequential(

            ### (B x 3 x 64 x 64) -> (B x 64 x 32 x 32) ###
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            ### (B x 3 x 64 x 64) -> (B x 64 x 32 x 32) ###
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            ### (B x 3 x 64 x 64) -> (B x 64 x 32 x 32) ###
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            ### (B x 3 x 64 x 64) -> (B x 64 x 32 x 32) ###
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            ### (B x 3 x 64 x 64) -> (B x 64 x 32 x 32) ###
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0),
        )

        self.apply(_init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.discriminator(x).reshape(batch_size,1)         

def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
