class BertLR(object):
    def __init__(self,optimizer,learning_rate,t_total,warmup):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self,training_step):
        lr_this_step = self.learning_rate * self.warmup_linear(training_step / self.t_total,self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step