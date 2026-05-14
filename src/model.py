import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.configuration_bert import BertConfig

class DNABERT2Baseline(nn.Module):
    def __init__(self, model_name: str, hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=config)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.GELU(),
            nn.Linear(64, 1),
        )

    def _masked_mean_pool(self, hidden_states, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean_pool(out[0], attention_mask)
        return self.classifier(pooled).squeeze(-1)