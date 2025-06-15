import torch
import torch.nn as nn
from transformers import LongformerPreTrainedModel, LongformerModel
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.energy = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.energy.weight)
        self.query.bias.data.zero_()
        self.energy.bias.data.zero_()

    def forward(self, hidden_states, attention_mask=None):
        # Compute attention scores
        transformed = torch.tanh(self.query(hidden_states))  # (batch_size, seq_len, hidden_dim)
        scores = self.energy(transformed).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Apply attention pooling
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        return pooled



class CustomLongformerForSequenceClassification(LongformerPreTrainedModel):
    """Longformer model with attention pooling for sequence classification.
    
    Uses attention pooling over the last four hidden layers instead of CLS token pooling.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Longformer backbone
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Attention pooling for each layer
        self.attention_poolers = nn.ModuleList([
            AttentionPooling(config.hidden_size) for _ in range(4)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)
        
        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Get last four hidden layers
        last_four_layers = outputs.hidden_states[-4:]
        
        # Apply attention pooling to each layer
        pooled = []
        for layer, pooler in zip(last_four_layers, self.attention_poolers):
            pooled.append(pooler(layer, attention_mask=attention_mask))
        
        # Concatenate pooled representations
        concatenated = torch.cat(pooled, dim=1)
        concatenated = self.dropout(concatenated)
        logits = self.classifier(concatenated)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if hasattr(self, 'loss_fct'):
                loss = self.loss_fct(logits, labels)
            else:
                loss = F.mse_loss(logits, labels.float())

        return {'loss': loss, 'logits': logits}

class CustomLongformerForSequenceClassification(LongformerPreTrainedModel):
    """Longformer model with attention pooling for sequence classification."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Longformer backbone
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Attention pooling for each layer
        self.attention_poolers = nn.ModuleList([
            AttentionPooling(config.hidden_size) for _ in range(4)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)
        
        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Get last four hidden layers
        last_four_layers = outputs.hidden_states[-4:]
        
        # Apply attention pooling to each layer
        pooled = []
        for layer, pooler in zip(last_four_layers, self.attention_poolers):
            pooled.append(pooler(layer, attention_mask=attention_mask))
        
        # Concatenate pooled representations
        concatenated = torch.cat(pooled, dim=1)
        concatenated = self.dropout(concatenated)
        logits = self.classifier(concatenated)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if hasattr(self, 'loss_fct'):
                loss = self.loss_fct(logits, labels)
            else:
                loss = F.mse_loss(logits.view(-1), labels.float().view(-1))

        return {'loss': loss, 'logits': logits}