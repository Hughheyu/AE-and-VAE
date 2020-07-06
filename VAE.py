import torch.nn as nn
import torch.nn.functional as F





'''
重构误差(二分类的交叉熵)
loss = -p*log(q) - (1-p)*log(1-q)
'''
reconstruction_function = nn.BCELoss(size_average=False)

def loss_function(recon_x,x,mu.logvar):
    BCE = reconstruction_function(recon_x,x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logavar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KLD_element = μ^2 + σ^2 - ln(σ^2) - 1
    return BCE+KLD   



class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(784,400)
        self.fc21 = nn.Linear(400,20)
        self.fc22 = nn.Linear(400,20)
        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(20,400)
        
    def encode(self,x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1) # mu, logvar
        
    def reparametrize(self,mu,logvar): 
        std = logvar.mul(0.5).exp_() # logvar = 2*ln(std)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()#采样eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) # std × eps + mu
        #return torch.randn(std.size()) * std + mu
        
    def decode(self,z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))
        
    def forward(self,x):
        mu,logvar = self.encoder(x)
        z = self.reparametrize(mu,logvar)
        return self.decode(z), mu, logvar