import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from src.models.basemodel import TorchModel
    
class EEGCT(TorchModel):
    def __init__(self, n_classes, n_features, n_heads, hidden_dim_1, hidden_dim_2, n_patches, n_time_points, device = 'cpu'):
        super().__init__(n_classes=n_classes, n_features=n_features, n_heads=n_heads, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, n_patches=n_patches, n_time_points=n_time_points, device=device)

    def set_model_parameters(self, n_classes, n_features, n_heads, hidden_dim_1, hidden_dim_2, n_patches, n_time_points):
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_heads = n_heads
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.n_patches = n_patches
        self.n_time_points = n_time_points
    
        self.lfe = self.LocalFeatureExtraction(self.n_features)

        self.ct = nn.Sequential(self.ConvTransformer(self.n_features, self.n_time_points, self.hidden_dim_1, self.n_heads),
                                self.ConvTransformer(self.n_features, self.n_time_points, self.hidden_dim_1, self.n_heads))

        self.conv2d1 = nn.Conv2d(self.n_patches, self.hidden_dim_2//2, (self.n_features, 3), stride = (1, 1), padding=(0, 1), bias=False)
        self.conv2d2 = nn.Conv2d(self.n_patches, self.hidden_dim_2//2, (self.n_features, 5), stride = (1, 1),  padding=(0, 2), bias=False)
        self.bn2d = nn.BatchNorm2d(self.hidden_dim_2)
        self.elu = nn.ELU()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim_2 * self.n_time_points, 500),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the EEGConvTransformer model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, m1, m2, n_time_points).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_classes).
        """

        # LOCAL FEATURE EXTRACTION
        x = self.lfe(x) # (batch_size, n_features, n_patches, n_time_points)

        # CONVOLUTIONAL TRANSFORMER BLOCKS
        x = self.ct(x) # (batch_size, n_features, n_patches, n_time_points)

        # CONVOLUTIONAL ENCODING
        x = torch.permute(x, (0, 2, 1, 3)) # (batch_size, n_patches, n_features, n_time_points)
        out = torch.cat((self.conv2d1(x), self.conv2d2(x)), axis = 1) # (batch_size, hidden_dim_2, 1, n_time_points)
        out = self.bn2d(out)
        out = self.elu(out)

        # CLASSIFICATION
        out = self.classifier(out)

        return out
    
    class LocalFeatureExtraction(nn.Module):
        """
        LocalFeatureExtraction module for extracting local features from input data.
        Args:
            n_features (int): Number of output features.
        Attributes:
            conv3d1 (nn.Conv3d): 3D convolutional layer with kernel size (8, 8, 3) and stride (4, 4, 1).
            conv3d2 (nn.Conv3d): 3D convolutional layer with kernel size (8, 8, 5) and stride (4, 4, 1).
            bn3d (nn.BatchNorm3d): Batch normalization layer.
            elu (nn.ELU): Exponential Linear Unit activation function.
        Methods:
            forward(x): Performs forward pass through the module.
        """

        def __init__(self, n_features):
            """
            Initializes the LocalFeatureExtraction class.
            Args:
                n_features (int): The number of features.
            Returns:
                None
            """

            super().__init__()
            self.n_features = n_features

            self.conv3d1 = nn.Conv3d(1, n_features//2, (8, 8, 3), stride=(4, 4, 1), padding = (0, 0, 1), bias=False)
            self.conv3d2 = nn.Conv3d(1, n_features//2, (8, 8, 5), stride=(4, 4, 1), padding = (0, 0, 2), bias=False)
            self.bn3d = nn.BatchNorm3d(n_features)
            self.elu = nn.ELU()

        def forward(self, x):
            """
            Performs forward pass through the LocalFeatureExtraction module.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 32, 32, n_time_points).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, n_features, n_patches, n_time_points).
            """

            x = x.unsqueeze(1) # (batch_size, 1, 32, 32, n_time_points)

            out = torch.cat((self.conv3d1(x), self.conv3d2(x)), dim = 1) # (batch_size, n_features, 7, 7, n_time_points)
            out = self.bn3d(out)
            out = self.elu(out)
            out = rearrange(out, 'b f h w t -> b f (h w) t') # (batch_size, n_features, n_patches, n_time_points)
            
            return out
        
    class ConvTransformer(nn.Module):
        """
        Convolutional Transformer module.
        Args:
            n_features (int): Number of input features.
            n_time_points (int): Number of time points.
            hidden_dim (int): Dimension of the hidden layer.
            n_heads (int): Number of attention heads.
        Attributes:
            mha (MultiHeadSelfAttention): Multi-head self-attention module.
            bn2d1 (nn.BatchNorm2d): Batch normalization layer.
            conv2d1 (nn.Conv2d): Convolutional layer 1.
            conv2d2 (nn.Conv2d): Convolutional layer 2.
            bn2d (nn.BatchNorm2d): Batch normalization layer.
            elu (nn.ELU): ELU activation function.
            conv2d3 (nn.Conv2d): Convolutional layer 3.
            bn2d2 (nn.BatchNorm2d): Batch normalization layer.
        Methods:
            forward(x): Forward pass of the ConvTransformer module.
        """

        def __init__(self, n_features, n_time_points, hidden_dim, n_heads):
            """
            Initializes the ConvTransformer class.
            Args:
                n_features (int): Number of input features.
                n_time_points (int): Number of time points.
                hidden_dim (int): Dimension of the hidden layer.
                n_heads (int): Number of attention heads.
            Returns:
                None
            """

            super().__init__()
            self.mha = self.MultiHeadSelfAttention(n_features, n_time_points, n_heads)
            self.bn2d1 = nn.BatchNorm2d(n_features)

            # self.cfe = convolutional_feature_expansion(C, E)
            self.conv2d1 = nn.Conv2d(n_features, hidden_dim//2, (1, 3), stride = (1, 1), padding=(0, 1), bias=False)
            self.conv2d2 = nn.Conv2d(n_features, hidden_dim//2, (1, 5), stride = (1, 1),  padding=(0, 2), bias=False)
            self.bn2d2 = nn.BatchNorm2d(hidden_dim)
            self.elu = nn.ELU()
            self.conv2d3 = nn.Conv2d(hidden_dim, n_features, (1, 1), stride = (1, 1), bias=False)
            self.bn2d3 = nn.BatchNorm2d(n_features)

        def forward(self, x):
            """
            Forward pass of the EEG-CT model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, n_features, n_patches, n_time_points).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, n_features, n_patches, n_time_points).
            """
            res = x
            ## Multi-head attention
            x = self.mha(x) # (batch_size, n_features, n_patches, n_time_points)

            x += res
            x = self.bn2d1(x)

            res = x

            ## Convolutional Feature Expansion
            x = torch.cat((self.conv2d1(x), self.conv2d2(x)), axis = 1) # (batch_size, hidden_dim, n_patches, n_time_points)
            x = self.bn2d2(x) 
            x = self.elu(x)
            x = self.conv2d3(x) # (batch_size, n_features, n_patches, n_time_points)

            x += res
            x = self.bn2d3(x)
            
            return x
        
        class MultiHeadSelfAttention(nn.Module):
            """
            Multi-head self-attention module.

            Args:
                n_features (int): Number of input features.
                n_time_points (int): Number of time points.
                n_heads (int): Number of attention heads.

            Attributes:
                n_features (int): Number of input features.
                n_time_points (int): Number of time points.
                n_heads (int): Number of attention heads.
                projections_per_head (int): Number of projections per head.
                conv (nn.Conv2d): Convolutional layer for computing Q, K, V.
                scale (float): Scaling factor for attention weights.

            Methods:
                forward(x): Forward pass of the module.

            """

            def __init__(self, n_features, n_time_points, n_heads):
                """
                Initializes the MultiHeadSelfAttention class.
                Args:
                    n_features (int): The number of input features.
                    n_time_points (int): The number of time points.
                    n_heads (int): The number of attention heads.
                Attributes:
                    n_features (int): The number of input features.
                    n_time_points (int): The number of time points.
                    n_heads (int): The number of attention heads.
                    projections_per_head (int): The number of projections per attention head.
                    conv (nn.Conv2d): The convolutional layer used for computing Q, K, V.
                    scale (float): The scaling factor used for attention scores.
                """

                super().__init__()
                self.n_features = n_features
                self.n_time_points = n_time_points
                self.n_heads = n_heads
                self.projections_per_head = n_features // n_heads

                self.qconv = nn.Sequential(nn.Conv2d(n_features,n_features, (1, 1), stride=(1, 1)),
                                          Rearrange('b (h d) p t -> b p (h d t)', h=self.n_heads))
                self.kconv = nn.Sequential(nn.Conv2d(n_features,n_features, (1, 1), stride=(1, 1)),
                                          Rearrange('b (h d) p t -> b p (h d t)', h=self.n_heads))
                self.vconv = nn.Sequential(nn.Conv2d(n_features,n_features, (1, 1), stride=(1, 1)),
                                          Rearrange('b (h d) p t -> b p (h d t)', h=self.n_heads))
                
                self.mha = nn.MultiheadAttention(n_features * n_time_points, n_heads, batch_first=True)

            def forward(self, x):
                """
                Forward pass of the multi-head self-attention module.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, n_features, n_patches, n_time_points).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, n_heads * projections_per_head, n_patches, n_time_points).

                """

                qp, kp, vp = self.qconv(x), self.kconv(x), self.vconv(x)
                
                attn_out, _ = self.mha(qp, kp, vp)
                attn_out = rearrange(attn_out, 'b p (d t) -> b d p t', t = self.n_time_points)

                return attn_out