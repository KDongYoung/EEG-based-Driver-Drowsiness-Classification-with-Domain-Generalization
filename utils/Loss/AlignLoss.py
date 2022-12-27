import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignLoss(nn.CrossEntropyLoss):
    """Alignment loss.
    Reference:
      Zhang et al. Robust domain-free domain generalization with class-aware alignment. ICASSP 2021.
    """
    def __init__(self, **args):
        super(AlignLoss, self).__init__()
        self.lossfn=torch.nn.CrossEntropyLoss()  
        self.weight = torch.tensor(args['align_weight'], device=args["device"])
           
    def forward(self, output, target):
        soft_label=F.softmax(output, dim=1)

        cross_entropy_loss = self.lossfn(output, target)
        
        # Calculate the Euclidean distance between the center and each soft label
        pdist=torch.nn.PairwiseDistance(p=2) 
        c=soft_label[target==0]
        ns_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
        c=soft_label[target==1]
        s_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
        align_loss=ns_align_loss+s_align_loss

        loss = cross_entropy_loss + self.weight*align_loss
        
        return loss