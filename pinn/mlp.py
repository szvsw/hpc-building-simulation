import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'device'
print(f"Using {device}")

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layer_ct=2, hidden_dim=128, output_dim=1, act=F.tanh, learnable_act=False, *args, **kwargs) -> None:
        super(MLP,self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.act = act
        self.learnable_act = learnable_act
        if learnable_act == "SINGLE":
            self.a = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        elif learnable_act == "MULTI":
            self.a = nn.Parameter(torch.ones(hidden_layer_ct+1)/(hidden_layer_ct + 1), requires_grad=True)
        else: 
            self.a = 1
        self.input_layer = nn.Sequential(nn.Linear(input_dim,hidden_dim))
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layer_ct):
            self.hidden_layers.append(nn.Linear(hidden_dim,hidden_dim))
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim,output_dim))
    
    def forward(self, x):
        x = self.input_layer(x)
        a = self.a[0] if self.learnable_act == "MULTI" else self.a
        x = self.act(a*x)

        skip = x
        for i,layer in enumerate(self.hidden_layers):
            x = layer(x)
            a = self.a[i+1] if self.learnable_act == "MULTI" else self.a
            x = self.act(a*x)
            if (i+1) % 2 == 0:
                x = skip + x
                skip = x

        x = self.output_layer(x)

        return x