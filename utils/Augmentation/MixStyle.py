import torch
import torch.nn as nn
import random

class MixStyle(nn.Module):
    """MixStyle in a multi-domain setting.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random',batch_size=1, num_domain=1):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

        self.batch_size=batch_size
        self.num_domain=num_domain
        
    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x, 0

        if random.random() > self.p:
            return x, 0

        return self.mixmixstyle(x)

    def mixmixstyle(self, x):
        B = x.size(0)
        mu = x.mean(dim=[2], keepdim=True) # channel dimension
        var = x.var(dim=[2], keepdim=True) # channel dimension
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B-1, -1, -1) 
            perm_b, perm_a = perm.chunk(2)
        
            if len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2+1)]
                perm_a = perm_a[torch.randperm(B // 2)]
            elif len(perm_a)<len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2+1)]
            elif len(perm_a)==len(perm_b):
                perm_b = perm_b[torch.randperm(B // 2)]
                perm_a = perm_a[torch.randperm(B // 2)]
            

            perm = torch.cat([perm_b, perm_a], 0)

        elif self.mix=="random_shift":
            # randomly shift the order of domains in the mini-batch
            shift_num=random.randint(1,self.num_domain-1)
            perm=torch.arange(0,B,dtype=torch.long)
            perm_a=perm[(-1)*self.batch_size*shift_num:]
            perm_b=perm[:(-1)*self.batch_size*shift_num]
            perm=torch.cat([perm_a,perm_b],0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix, 1
