import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

'''
# This is the official ProxyAnchor loss.

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        #self.A = torch.nn.Parameter(torch.randn(180, 180).cuda())
        #nn.init.kaiming_normal_(self.A, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        # P.shape = torch.Size([100, 512]), because proxies=100 and embedding_size=512

        # this is the same as doing l2_norm(X).matmul(l2_norm(P).T)
        cos_batch_cnt = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        #cos = F.linear(l2_norm(X), l2_norm(X))

        #print(self.A)
        #print(self.A[0])
        #temp = cos_batch_cnt.T.matmul(self.A)
        cos_cnt_cnt = cos_batch_cnt.T.matmul(cos_batch_cnt)

        #import pdb
        #pdb.set_trace()

        #cos_cnt_cnt = F.linear(l2_norm(P), l2_norm(P))

        #print(cos_cnt_cnt[0])
        # cos.shape = torch.Size([180, 100]), because batch_size=180 and proxies=100
        # the P_one_hot are the labels in one-hot encoding (so they are the positives)
        # P_one_hot.shape = torch.Size([180, 100]), because batch_size=180 and labels=100
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        #P_one_hot = torch.eye(self.nb_classes, self.nb_classes).cuda()
        #N_one_hot = torch.ones(self.nb_classes, self.nb_classes).cuda() - torch.eye(self.nb_classes, self.nb_classes).cuda()

        N_one_hot_2 = 1 - P_one_hot
        
        #pos_exp = torch.exp(-self.alpha * (cos_batch_cnt - self.mrg))
        pos_exp = torch.exp(-self.alpha * (cos_batch_cnt - self.mrg))
        #neg_exp = torch.exp(self.alpha * (cos_cnt_cnt + self.mrg))

        neg_exp_2 = torch.exp(self.alpha * (cos_batch_cnt + self.mrg))

        # this one finds the proxies that have positives in batch
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        #N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        N_sim_sum_2 = torch.where(N_one_hot_2 == 1, neg_exp_2, torch.zeros_like(neg_exp_2)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        #neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        neg_term_2 = torch.log(1 + N_sim_sum_2).sum() / self.nb_classes

        loss = pos_term + neg_term

        #import pdb
        #pdb.set_trace()
        
        return loss

'''

#'''
class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        #self.A = torch.nn.Parameter(torch.randn(180, 180).cuda())
        #nn.init.kaiming_normal_(self.A, mode='fan_out')
        
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        # P.shape = torch.Size([100, 512]), because proxies=100 and embedding_size=512

        # this is the same as doing l2_norm(X).matmul(l2_norm(P).T)
        cos_batch_cnt = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        #cos = F.linear(l2_norm(X), l2_norm(X))

        #print(self.A)
        #print(self.A[0])
        #temp = cos_batch_cnt.T.matmul(self.A)
        cos_cnt_cnt = cos_batch_cnt.T.matmul(cos_batch_cnt)

        #import pdb
        #pdb.set_trace()

        #cos_cnt_cnt = F.linear(l2_norm(P), l2_norm(P))

        #print(cos_cnt_cnt[0])
        # cos.shape = torch.Size([180, 100]), because batch_size=180 and proxies=100
        # the P_one_hot are the labels in one-hot encoding (so they are the positives)
        # P_one_hot.shape = torch.Size([180, 100]), because batch_size=180 and labels=100
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        #P_one_hot = torch.eye(self.nb_classes, self.nb_classes).cuda()
        N_one_hot = torch.ones(self.nb_classes, self.nb_classes).cuda() - torch.eye(self.nb_classes, self.nb_classes).cuda()

        #N_one_hot_2 = 1 - P_one_hot
        
        #pos_exp = torch.exp(-self.alpha * (cos_batch_cnt - self.mrg))
        pos_exp = torch.exp(-self.alpha * (cos_batch_cnt - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos_cnt_cnt + self.mrg))

        #neg_exp_2 = torch.exp(self.alpha * (cos_batch_cnt + self.mrg))

        # this one finds the proxies that have positives in batch
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        #N_sim_sum_2 = torch.where(N_one_hot_2 == 1, neg_exp_2, torch.zeros_like(neg_exp_2)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        #neg_term_2 = torch.log(1 + N_sim_sum_2).sum() / self.nb_classes

        loss = pos_term + neg_term

        #import pdb
        #pdb.set_trace()
        
        return loss

#'''
'''
class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        #with torch.no_grad():
        #   self.proxies = torch.load('proxies/proxies.pt')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = torch.nn.Parameter(torch.tensor(1.0).cuda(), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(1.0).cuda(), requires_grad=True)
        
    def forward(self, X, T):
        P = self.proxies
        
        with open('alphas.csv', 'a') as file:
            file.write(str(float(self.alpha)) + '\n')

        # P.shape = torch.Size([100, 512]), because proxies=100 and embedding_size=512
        #import pdb
        #pdb.set_trace()
        # this is the same as doing l2_norm(X).matmul(l2_norm(P).T)
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        # cos.shape = torch.Size([180, 100]), because batch_size=180 and proxies=100
        # the P_one_hot are the labels in one-hot encoding (so they are the positives)
        # P_one_hot.shape = torch.Size([180, 100]), because batch_size=180 and labels=100
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        #neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        neg_exp = torch.exp(self.beta * (cos + self.mrg))

        # this one finds the proxies that have positives in batch
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        #loss = neg_term

        return loss
'''
# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed,  softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, thresh = 0.5, epsilon = 0.1, scale_pos = 2, scale_neg = 50):
        super(MultiSimilarityLoss, self).__init__()
        #self.thresh = 0.5
        #self.epsilon = 0.1
        #self.scale_pos = 2
        #self.scale_neg = 50

        self.thresh = 0.77
        self.epsilon = 0.39
        self.scale_pos = 17.97
        self.scale_neg = 75.66
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, nb_classes, sz_embed, margin=28.6, scale=220, **kwargs):
        super(ArcFaceLoss, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.margin = margin
        self.scale = scale
        self.loss_func = losses.ArcFaceLoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, margin = self.margin, scale = self.scale)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
'''
class SoftTripleLoss(nn.Module):
    def __init__(self, nb_classes, sz_embed, centers_per_class=10, la=20, gamma=0.1, margin=0.01, **kwargs):
        super(SoftTripleLoss, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.centers_per_class = centers_per_class
        self.la = la
        self.gamma = gamma
        self.margin = margin
        self.fc = torch.nn.Parameter(torch.Tensor(self.sz_embed, self.nb_classes * centers_per_class))
        #self.weight_init_func(self.fc)
        self.loss_func = losses.SoftTripleLoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, centers_per_class = self.centers_per_class, la = self.la, gamma = self.gamma, margin = self.margin)

        def forward(self, embeddings, labels):
            loss = self.loss_func(embeddings, labels)
            return loss
'''
class SoftTripleLoss(nn.Module):
    def __init__(self, margin, lamda , gamma, tau, nb_classes, sz_embed):
        super(SoftTripleLoss, self).__init__()
        
        # Default values
        self.lamda = lamda
        self.gamma = gamma
        self.tau = tau
        self.margin = margin
        
        # Values from searching
        #self.margin = 0.4
        #self.lamda = 78
        #self.gamma = 58
        #self.tau = 0.4
        
        self.centers_per_class = 2
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.fc = torch.nn.parameter.Parameter(torch.Tensor(self.sz_embed, self.nb_classes * self.centers_per_class))
        self.weight = torch.zeros(self.nb_classes*self.centers_per_class, self.nb_classes*self.centers_per_class, dtype=torch.bool).cuda()

        for i in range(0, self.nb_classes):
            for j in range(0, self.centers_per_class):
                self.weight[i*self.centers_per_class+j, i*self.centers_per_class+j+1:(i+1)*self.centers_per_class] = 1
        torch.nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))

        return

    def forward(self, embeddings, labels):
        centers = torch.nn.functional.normalize(self.fc, p=2, dim=0)
        centers = centers.cuda()

        simInd = embeddings.matmul(centers)
        simStruc = simInd.reshape(-1, self.nb_classes, self.centers_per_class)
        prob = torch.nn.functional.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)

        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), labels] = self.margin

        labels = labels.cuda()
        
        loss = torch.nn.functional.cross_entropy(self.lamda*(simClass-marginM), labels)

        if self.tau > 0 and self.centers_per_class > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.nb_classes*self.centers_per_class*(self.centers_per_class-1.))

            return loss+self.tau*reg

        else:

            return loss
'''
class LiftedStructureLoss(nn.Module):
    def __init__(self, neg_margin=1, pos_margin=0, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
        self.loss_func = losses.GeneralizedLiftedStructureLoss(neg_margin = self.neg_margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
'''

class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=40, beta=2, margin=0.5, hard_mining=True,  **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        n = embeddings.size(0)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        labels = labels
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], labels==labels[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], labels!=labels[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:

                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
            
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue 
            
                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair)))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair)))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)
        loss = sum(loss)/n

        return loss

class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, **kwargs):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.nu = nu
        self.beta = beta
        self.miner = miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)
        self.loss_func = losses.MarginLoss(margin = self.margin, nu = self.nu, beta = self.beta)

    def forward(self, embeddings, labels):
    	distance_weighted_pairs = self.miner(embeddings, labels)
    	loss = self.loss_func(embeddings, labels, distance_weighted_pairs)
    	return loss

