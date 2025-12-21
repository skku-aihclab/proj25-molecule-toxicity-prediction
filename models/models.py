import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torchvision import models
from transformers import AutoModel, AutoTokenizer

class GraphEncoder(nn.Module):
    """ 
    Graph encoder based on the Graph Isomorphism Network (GIN), a GNN architecture
    tailored for molecular graph analysis.
    This module consists of multiple GINEConv layers followed by a projection head
    that reduces the output dimension to `emb_dim`. Each GINEConv layer applies
    a multi-layer perceptron (MLP) to the node features and aggregates information
    from neighboring nodes.
    Reference: "How Powerful are Graph Neural Networks?"
    Args:
        in_dim (int): dimension of input node features 
        hidden_dim (int): dimension of hidden layers
        num_layers (int): number of GINEConv layers
        emb_dim (int): dimension of output embeddings
        dropout (float): dropout rate
    """
    def __init__(
        self,
        in_dim: int = 78,
        hidden_dim: int = 256,
        num_layers: int = 3,
        emb_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # define GINEConv layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=4))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # define 1-layer MLP head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),   
        )
        self.emb_dim = emb_dim
        self.dropout = dropout

    def forward(self, data):
        # Extract node features, edge indices, edge attributes, and batch information
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch) # Global pooling (sum)
        x = self.proj(x) 
        return x

class GraphClassifier(nn.Module):
    """
    GraphClassifier takes the output of GraphEncoder and applies a linear layer
    to produce multi-task logits. The output dimension is determined by `num_tasks`.
    """
    def __init__(self,
                 encoder: GraphEncoder,
                 num_tasks: int = 12):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(
            encoder.emb_dim,
            num_tasks
        )

    def forward(self, data):
        emb    = self.encoder(data)
        return self.classifier(emb)
    
class SMILESEncoder(nn.Module):
    """
    SMILESEncoder module uses a pre-trained Transformer (MoLFormer-XL) 
    to encode molecular SMILES strings into fixed-size embeddings.
    It includes a 2-layer MLP head for further processing.
    Reference: "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties"
    Args:
        pretrained_model_name (str): Name of the pre-trained model to use.
        emb_dim (int): Dimension of the output embeddings.
        dropout (float): Dropout rate for the MLP layers.
        max_length (int): Maximum length of input SMILES strings.
    """
    def __init__(
        self,
        pretrained_model_name: str = "ibm-research/MoLFormer-XL-both-10pct",
        emb_dim: int = 128,
        dropout: float = 0.1,
        max_length: int = 202,
    ):
        super().__init__()
        # Load pre-trained model (MoLFormer-XL)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        self.max_length = max_length
        self.emb_dim = emb_dim
        hidden_size = self.encoder.config.hidden_size  # output dimension of molformer

        # unfreeze the model for training
        for p in self.encoder.parameters():
            p.requires_grad = True

        # define 2-layer MLP head
        # Use GELU instead of ReLU as recommended by the original paper
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, emb_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        emb = out.last_hidden_state.mean(dim=1) 

        h1 = self.proj[0](emb)  
        h1 = self.proj[1](h1)  
        h1 = self.proj[2](h1) 
        h1 = h1 + emb            # Residual connection
        h2 = self.proj[3](h1)    
        h2 = self.proj[4](h2)  
        h2 = self.proj[5](h2) 

        return h2 

class SMILESClassifier(nn.Module):
    """
    SMILESClassifier takes the output of SMILESEncoder and applies a linear layer
    to produce multi-task logits. The output dimension is determined by `num_tasks`.
    """
    def __init__(self, encoder: SMILESEncoder, num_tasks: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.emb_dim, num_tasks)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(input_ids, attention_mask) 
        return self.classifier(emb) 


class ImageEncoder(nn.Module):
    """
    ImageEncoder using pre-trained ResNet18 (e.g., ImageMol) backbone for molecular images.
    This module takes images as input and outputs embeddings of a specified dimension.
    It includes a 2-layer MLP head for further processing.
    Reference: "Accurate prediction of molecular properties and drug targets using a self-supervised image representation learning framework"
    Args:
        ckpt_path (str): Path to the pre-trained model checkpoint.
        dropout (float): Dropout rate for the MLP layers.
        hidden_size (int): Hidden size for the MLP layers.
        emb_dim (int): Output embedding dimension.
    """
    def __init__(
        self, 
        ckpt_path: str, 
        dropout: float = 0.1, 
        hidden_size: int = 256, 
        emb_dim: int = 128
    ):

        super().__init__()

        # Define the backbone using ResNet18
        self.backbone = models.resnet18(pretrained=False)

        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Load pre-trained ImageMol weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        # Remove 'module.' prefix from keys
        clean_state = {}
        for k, v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            clean_state[new_key] = v

        # remove 'fc.' prefix from keys
        backbone_state = {
            k: v for k, v in clean_state.items() if not k.startswith("fc.")
        }
        self.backbone.load_state_dict(backbone_state, strict=False)
        
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size

        # unfreeze the model(ImageMol) for training
        for p in self.backbone.parameters():
            p.requires_grad = True 

        # define 2-layer MLP head
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.mlp(feat)
        return emb



class ImageClassifier(nn.Module):
    """
    ImageClassifier takes the output of ImageEncoder and applies a linear layer
    to produce multi-task logits. The output dimension is determined by `num_tasks`.
    """
    def __init__(self, encoder: ImageEncoder, num_tasks: int = 12):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.emb_dim, num_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.classifier(emb)

    
class SpectrumEncoder(nn.Module):
    """
    SpectrumEncoder using pre-trained CReSS NMR encoder to process spectral data
    This module takes a list of ppm values as input and outputs embeddings of a specified dimension.
    It includes a 2-layer MLP head for further processing.
    Reference: "Cross-Modal Retrieval between 13C NMR Spectra and Structures for Compound Identification Using Deep Contrastive Learning"
    Args:
        model_inference: CReSS NMR encoder instance
        hidden_dim (int): Dimension of the hidden layer in the MLP.
        emb_dim (int): Dimension of the output embeddings.
        dropout (float): Dropout rate for the MLP layers.
    """

    def __init__(
        self,
        model_inference,
        hidden_dim: int = 512,
        emb_dim: int = 256,
        dropout: float = 0.5
    ):

        super().__init__()
        self.model_inf = model_inference

        # Freeze the CReSS NMR model for training
        for p in self.model_inf.clip_model.parameters():
            p.requires_grad = False 

        # Learned embedding for missing spectrum data
        # Initialize with zeros to represent "no information" state
        # Alows the model to learn a representation when spectrum is unavailable
        self.missing_token = nn.Parameter(torch.zeros(1, emb_dim))

        # define 2-layer MLP head
        self.fc1 = nn.Linear(768, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.emb_dim = emb_dim


    def forward(self, ppm_lists):
        embeddings = []
        # create embeddings for each ppm list
        for ppm in ppm_lists:
            if ppm is None:
                # Use learned missing token for samples without spectrum data
                emb_processed = self.missing_token
            else:
                # Process normal spectrum data through CReSS encoder
                emb = self.model_inf.nmr_encode(ppm)
                # Apply MLP to transform to desired dimension
                h = self.fc1(emb)
                h = self.relu1(h)
                h = self.dropout(h)
                h = self.fc2(h)
                h = self.relu2(h)
                emb_processed = self.dropout(h)

            embeddings.append(emb_processed)

        # Stack all embeddings (batch_size, emb_dim)
        batch_emb = torch.cat(embeddings, dim=0)

        return batch_emb

class SpectrumClassifier(nn.Module):
    """
    SpectrumClassifier takes the output of SpectrumEncoder and applies a linear layer
    to produce multi-task logits. The output dimension is determined by `num_tasks`.
    """

    def __init__(self, encoder: SpectrumEncoder, num_tasks: int = 12):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = encoder.emb_dim
        self.classifier = nn.Linear(self.emb_dim, num_tasks)

    def forward(self, ppm_lists):
        emb = self.encoder(ppm_lists)  
        return self.classifier(emb) 
    
class MultiModalGphSMI(nn.Module):
    """
    MultiModalGphSMI: combines Graph and SMILES branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        gph_encoder (nn.Module): GNN based graph encoder.
        smi_encoder (nn.Module): Transformer based SMILES encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
    """
    def __init__(
        self,
        gph_encoder: nn.Module,
        smi_encoder: nn.Module,
        emb_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 512,
    num_layers: int = 1,
        num_tasks: int = 12,
        dropout: float = 0.3
    ):
        super().__init__()
        self.gph_encoder = gph_encoder
        self.smi_encoder = smi_encoder

        self.proj_g = nn.Linear(gph_encoder.emb_dim, emb_dim)
        self.proj_s = nn.Linear(smi_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        fuse_in = emb_dim

        # Define the fusion MLP layers
        layers = []
        in_dim = fuse_in
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, graph, ids, mask, return_attention=False):
        # ── Graph branch ──────────────────────────────
        g = F.relu(self.proj_g(self.gph_encoder(graph)))

        # ── SMILES branch ──────────────────────
        s = F.relu(self.proj_s(self.smi_encoder(ids, mask)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([g, s], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ─────────────────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits
    
class MultiModalGphImg(nn.Module):
    """
    MultiModalGphImg combines Graph and Image branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        gph_encoder (nn.Module): GNN based graph encoder.
        img_encoder (nn.Module): 2D CNN based image encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 gph_encoder: nn.Module,
                 img_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.gph_encoder = gph_encoder
        self.img_encoder = img_encoder

        self.proj_g = nn.Linear(gph_encoder.emb_dim, emb_dim)
        self.proj_i = nn.Linear(img_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, graph, img, return_attention=False):
        # ── Graph branch ──────────────────────────────
        g = F.relu(self.proj_g(self.gph_encoder(graph)))

        # ── Image branch ──────────────────────────────
        i = F.relu(self.proj_i(self.img_encoder(img)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([g, i], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits
    
class MultiModalGphSpec(nn.Module):
    """
    MultiModalGphSpec: combines Graph and Spectrum branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        gph_encoder (nn.Module): GNN based graph encoder.
        img_encoder (nn.Module): CNN1D based spectrum encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the head MLP.
    """
    def __init__(self,
                 gph_encoder: nn.Module,
                 spec_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.gph_encoder = gph_encoder
        self.spec_encoder = spec_encoder

        self.proj_g = nn.Linear(gph_encoder.emb_dim, emb_dim)
        self.proj_sp = nn.Linear(spec_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, graph, ppm_lists, return_attention=False):
        # ── Graph branch ──────────────────────────────
        g = F.relu(self.proj_g(self.gph_encoder(graph)))

        # ── Spectrum branch ──────────────────────────────
        sp = F.relu(self.proj_sp(self.spec_encoder(ppm_lists)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([g, sp], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ─────────────────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits
    
class MultiModalGphSMIImg(nn.Module):
    """
    MultiModalGphSMIImg combines Graph, SMILES, Image branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        gph_encoder (nn.Module): GNN based graph encoder.
        smi_encoder (nn.Module): Transformer based SMILES encoder.
        img_encoder (nn.Module): 2D CNN based image encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 gph_encoder: nn.Module,
                 smi_encoder: nn.Module,
                 img_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.gph_encoder = gph_encoder
        self.smi_encoder = smi_encoder
        self.img_encoder = img_encoder
        self.proj_g = nn.Linear(gph_encoder.emb_dim, emb_dim)
        self.proj_s = nn.Linear(smi_encoder.emb_dim, emb_dim)
        self.proj_i = nn.Linear(img_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, graph, ids, mask, img, return_attention=False):
        # ── Graph branch ──────────────────────────────
        g = F.relu(self.proj_g(self.gph_encoder(graph)))

        # ── SMILES branch ──────────────────────
        s = F.relu(self.proj_s(self.smi_encoder(ids, mask)))

        # ── Image branch ──────────────────────────────
        i = F.relu(self.proj_i(self.img_encoder(img)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([g, s, i], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits
    
class MultiModalSMIImg(nn.Module):
    """
    MultiModalSMIImg combines SMILES and Image branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        smi_encoder (nn.Module): Transformer based SMILES encoder.
        img_encoder (nn.Module): 2D CNN based image encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 smi_encoder: nn.Module,
                 img_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.smi_encoder = smi_encoder
        self.img_encoder = img_encoder

        self.proj_s = nn.Linear(smi_encoder.emb_dim, emb_dim)
        self.proj_i = nn.Linear(img_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, ids, mask, img, return_attention=False):
        # ── SMILES branch ──────────────────────
        s = F.relu(self.proj_s(self.smi_encoder(ids, mask)))

        # ── Image branch ──────────────────────────────
        i = F.relu(self.proj_i(self.img_encoder(img)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([s, i], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits

class MultiModalSMISpec(nn.Module):
    """
    MultiModalSMISpec combines SMILES and Spectrum branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        smi_encoder (nn.Module): Transformer based SMILES encoder.
        spec_encoder (nn.Module): 1D CNN based spectrum encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 smi_encoder: nn.Module,
                 spec_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.smi_encoder = smi_encoder
        self.spec_encoder = spec_encoder

        self.proj_s = nn.Linear(smi_encoder.emb_dim, emb_dim)
        self.proj_sp = nn.Linear(spec_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, ids, mask, spec, return_attention=False):
        # ── SMILES branch ──────────────────────
        s = F.relu(self.proj_s(self.smi_encoder(ids, mask)))

        # ── Spectrum branch ──────────────────────────────
        sp = F.relu(self.proj_sp(self.spec_encoder(spec)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([s, sp], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits

class MultiModalSpecImg(nn.Module):
    """
    MultiModalSpecImg combines Spectrum and Image branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        spec_encoder (nn.Module): 1D CNN based spectrum encoder.
        img_encoder (nn.Module): 2D CNN based image encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 spec_encoder: nn.Module,
                 img_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        self.spec_encoder = spec_encoder
        self.img_encoder = img_encoder

        self.proj_sp = nn.Linear(spec_encoder.emb_dim, emb_dim)
        self.proj_i = nn.Linear(img_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, spec, img, return_attention=False):
        # ── Spectrum branch ──────────────────────────────
        sp = F.relu(self.proj_sp(self.spec_encoder(spec)))

        # ── Image branch ──────────────────────────────
        i = F.relu(self.proj_i(self.img_encoder(img)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([sp, i], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits

class MoltiTox(nn.Module):
    """
    MoltiTox combines Graph, Image, SMILES, and Spectrum branches for multimodal learning.
    It projects the outputs of each branch to a common embedding dimension and fuses them using self-attention.
    The fused representation is then passed through a classification head to produce multi-task logits.
    Args:
        gph_encoder (nn.Module): GNN based graph encoder.
        smi_encoder (nn.Module): Transformer based SMILES encoder.
        img_encoder (nn.Module): 2D CNN based image encoder.
        spec_encoder (nn.Module): 1D CNN based spectrum encoder.
        emb_dim (int): Dimension of the output embeddings for each branch.
        num_heads (int): Number of attention heads for multi-head self-attention.
        hidden_dim (int): Hidden size of each layer in the head MLP.
        num_layers (int): Number of layers in the head MLP.
        num_tasks (int): Number of tasks for multi-task learning.
        dropout (float): Dropout rate for the fusion layers.
    """
    def __init__(self,
                 gph_encoder: nn.Module,
                 smi_encoder: nn.Module,
                 img_encoder: nn.Module,
                 spec_encoder: nn.Module,
                 emb_dim: int = 128,
                 num_heads: int = 4,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 num_tasks: int = 12,
                 dropout: float = 0.3):
        super().__init__()
        # Backbone encoders
        self.gph_encoder = gph_encoder
        self.smi_encoder = smi_encoder
        self.img_encoder = img_encoder
        self.spec_encoder = spec_encoder

        # Project each modality to common dimension
        self.proj_g = nn.Linear(gph_encoder.emb_dim, emb_dim)
        self.proj_s = nn.Linear(smi_encoder.emb_dim, emb_dim)
        self.proj_i = nn.Linear(img_encoder.emb_dim, emb_dim)
        self.proj_sp = nn.Linear(spec_encoder.emb_dim, emb_dim)

        # Multi-head self-attention over modalities
        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define the fusion MLP layers
        layers = []
        in_dim = emb_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_tasks))
        self.head = nn.Sequential(*layers)

    def forward(self, graph, ids, mask, img, spec, return_attention=False):
        # ── Graph branch ──────────────────────────────
        g = F.relu(self.proj_g(self.gph_encoder(graph)))

        # ── SMILES branch ──────────────────────
        s = F.relu(self.proj_s(self.smi_encoder(ids, mask)))

        # ── Image branch ──────────────────────────────
        i = F.relu(self.proj_i(self.img_encoder(img)))

        # ── Spectrum branch ──────────────────────────────
        sp = F.relu(self.proj_sp(self.spec_encoder(spec)))

        # ── Stack features ─────────────────────────────
        modal_feats = torch.stack([g, s, i, sp], dim=1)

        # ── Multi-head self-attention fusion ─────────────
        attn_out, attn_weights = self.mha(modal_feats, modal_feats, modal_feats)

        # ── Pool across modality tokens ─────────────────
        fused = attn_out.mean(dim=1)

        # ── Classification head ───────────────────────────
        logits = self.head(fused)

        if return_attention:
            return logits, attn_weights
        return logits