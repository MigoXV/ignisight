from fairseq.models import BaseFairseqModel ,register_model_architecture,register_model
import torch
from torch import nn

@register_model('bert')
class BertModel(BaseFairseqModel):
    @classmethod
    def build_model(cls,args,task):
        return cls()
    
    def __init__(self, max_seq_len=512, d_model=384, num_layers=2, num_heads=6, d_ff=768, dropout=0.2):
            super().__init__()
            # self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(dropout)
            
            # Encoder Layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True  # Pre-LN 结构，训练更稳定
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 分类头（示例）
            self.fc = nn.Linear(d_model, 6)

    def forward(self, x:torch.Tensor):
        x = x.mean(dim=1).transpose(1,2)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # token_emb = self.embedding(x)
        pos_emb = self.pos_embedding(positions)
        x = self.dropout(x + pos_emb)
        
        # Transformer Encoder
        x = self.encoder(x)  # [batch_size, seq_len, d_model]
        
        cls_output = x.mean(dim=1)  # 平均池化
        return self.fc(cls_output)
    
@register_model_architecture('bert', 'tiny_bert')
def register_bert(model):
    pass