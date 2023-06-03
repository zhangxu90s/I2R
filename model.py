import torch.nn as nn
import torch    
import typing as tp
import torch.nn.functional as F

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        
    def forward(self, code_inputs=None, nl_inputs=None): 
        p_dpr = 0.4

        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
            goutputs = (outputs[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]

            # -- Set models dropout --#\

            if p_dpr != 0.1:
                self.encoder = self.set_dropout_mf(self.encoder, w=p_dpr)
                logits = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))
                logits = (logits[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            if p_dpr != 0.1: self.encoder = self.set_dropout_mf(self.encoder, w=0.1)

            return torch.nn.functional.normalize(logits, p=2, dim=1),torch.nn.functional.normalize(goutputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
            goutputs = (outputs[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]

            # -- Set models dropout --#

            if p_dpr != 0.1:
                self.encoder = self.set_dropout_mf(self.encoder, w=p_dpr)
                logits = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
                logits = (logits[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            
            if p_dpr != 0.1: self.encoder = self.set_dropout_mf(self.encoder, w=0.1)

            return torch.nn.functional.normalize(logits, p=2, dim=1),torch.nn.functional.normalize(goutputs, p=2, dim=1)    
    
    def set_dropout_mf(
        self, 
        model:nn, 
        w:tp.List[float]
        ):
        """Alters the dropouts in the embeddings.
        """
        # ------ set hidden dropout -------#
        if hasattr(model, 'module'):
            model.module.embeddings.dropout.p = w
            for i in model.module.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w        
        else:
            model.embeddings.dropout.p = w
            for i in model.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w
            
        return model
    