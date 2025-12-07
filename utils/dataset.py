import os
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset as GeometricDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
from typing import List
from PIL import Image



""" Define atom feature lists for one-hot encoding (total: 78) """
ATOM_LISTS = {
    # 1) Atom symbols (one-hot, 44 dims)
    "atom_symbol": [
        'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al',
        'I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn',
        'H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb',
        'Unknown'
    ],
    # 2) Degree (one-hot, 11 dims)
    "degree": list(range(0, 11)),
    # 3) Implicit valence (one-hot, 7 dims)
    "implicit_valence": list(range(0, 7)),
    # 4) Hybridization (one-hot, 5 dims)
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
    # 5) Total hydrogens (one-hot, 5 dims)
    "total_hydrogens": list(range(0, 5)),
    # 6) Aromatic (scalar boolean, 1 dim)
    "is_aromatic": [False, True],
    # 7) Formal charge (scalar integer, 1 dim)
    "formal_charge": list(range(-3, 4)),
    # 8) Radical electrons (scalar integer, 1 dim)
    "radical_electrons": list(range(0, 3)),
    # 9) Chirality code (one-hot R/S, 2 dims)
    "chirality_code": ['R', 'S'],
    # 10) Chirality possible (scalar boolean, 1 dim)
    "chirality_possible": [False, True],
}

def atom_to_feature_vector(atom):
    feats = []
    # 1) atom_symbol (one-hot, 44 dims)
    feats += [1 if atom.GetSymbol() == sym else 0
              for sym in ATOM_LISTS["atom_symbol"]]
    # 2) degree (one-hot, 11 dims)
    feats += [1 if atom.GetDegree() == d else 0
              for d in ATOM_LISTS["degree"]]
    # 3) implicit_valence (one-hot, 7 dims)
    feats += [1 if atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)  == v else 0
              for v in ATOM_LISTS["implicit_valence"]]
    # 4) hybridization (one-hot, 5 dims)
    feats += [1 if atom.GetHybridization() == h else 0
              for h in ATOM_LISTS["hybridization"]]
    # 5) total_hydrogens (one-hot, 5 dims)
    feats += [1 if atom.GetTotalNumHs() == h else 0
              for h in ATOM_LISTS["total_hydrogens"]]
    # 6) is_aromatic (scalar boolean, 1 dim)
    feats.append(int(atom.GetIsAromatic()))
    # 7) formal_charge (scalar integer, 1 dim)
    feats.append(atom.GetFormalCharge())
    # 8) radical_electrons (scalar integer, 1 dim)
    feats.append(atom.GetNumRadicalElectrons())
    # 9) chirality_code (one-hot, 2 dims: 'R','S')
    cip = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None
    feats += [1 if cip == code else 0
              for code in ATOM_LISTS["chirality_code"]]
    # 10) chirality_possible (scalar boolean, 1 dim)
    feats.append(int(atom.HasProp('_CIPCode')))
    # convert to tensor
    return torch.tensor(feats, dtype=torch.float)

def smiles_to_graph(smiles):
    """ Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # create node features
    x = torch.stack([atom_to_feature_vector(a) for a in mol.GetAtoms()])

    # Prepare lists to collect edge indices and edge attributes
    edge_list = []
    edge_attr_list = []

    # Iterate over each bond in the molecule
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bond_feature = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
        # Add both directions for undirected graph representation
        edge_list += [[i, j], [j, i]]
        edge_attr_list += [bond_feature, bond_feature]

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() 
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)     
    else:
        # Handle cases where no bonds are detected
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 4), dtype=torch.float)

    # Return the PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GraphDataset(GeometricDataset):
    """
    Convert SMILES strings to graph data for PyTorch Geometric.
    Args:
        csv_path (str): Path to the CSV file containing SMILES and labels.
        label_name (List[str]): List of label column names in the CSV.
    Returns:
        data: PyTorch Geometric Data object with:
            - Node feature matrix of shape (num_nodes, num_node_features)
            - Edge index tensor of shape (2, num_edges)
            - Edge feature matrix of shape (num_edges, num_edge_features)
            - label
    """
    def __init__(self, csv_path, label_name:List[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.smiles = self.df['smiles'].tolist()
        self.labels = self.df[label_name].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        data = smiles_to_graph(smi)  # Convert SMILES to graph
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        data.y = y.unsqueeze(0) 
        return data
    
class SMILESDataset(Dataset):
    """
    Convert SMILES strings to tokenized input for Transformer models.
    Args:
        csv_path (str): Path to the CSV file containing SMILES and labels.
        label_name (str): Name of the label column in the CSV.
        pretrained_model_name (str): Name of the pretrained model for tokenization.
        max_length (int): Maximum sequence length for tokenization.
    Returns:
        input_ids: Tensor containing token IDs.
        attention_mask: Tensor indicating padded tokens.
        label
    """
    def __init__(self, csv_path, label_name, pretrained_model_name= "ibm-research/MoLFormer-XL-both-10pct",max_length= 202):
        self.df = pd.read_csv(csv_path)
        self.smiles = self.df['smiles'].tolist()
        self.labels = self.df[label_name].values.astype(float)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        encoding = self.tokenizer(
            smi,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # convert 2D tensors to 1D
        input_ids   = encoding['input_ids'].squeeze(0)   
        attn_mask   = encoding['attention_mask'].squeeze(0) 
        label       = torch.tensor(self.labels[idx], dtype=torch.float)
        return input_ids, attn_mask, label

class ImageDataset(Dataset):
    """
    Convert images to RGB tensors for CNN 2D model.
    Args:
        csv_path (str): Path to the CSV file containing metadata.
        label_name (str): Name of the label column in the CSV.
        img_dir (str): Directory containing the image files.
        transform (optional): transform can be applied to the images (e.g., Normalize).
    Returns:
        image: RGB image tensor of shape (3, H, W)
        label
    """
    def __init__(self, csv_path, label_name, img_dir, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.label_name = label_name
        self.labels = self.df[label_name].values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mol_id = self.df.loc[idx, 'mol_id'] # load image by mol_id
        img_path = os.path.join(self.img_dir, f"{mol_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    
    

class SpectrumDataset(Dataset):
    """
    Convert spectral data from .npy files for CNN 1D model.
    Args:
        csv_path (str): Path to the CSV file containing metadata.
        label_name (List[str]): List of label column names in the CSV.
        spectra_dir (str): Directory containing the spectral .npy files.
        allow_missing (bool): If True, allows samples without spectrum files and returns None for missing spectra.
    Returns:
        spec: Spectral data tensor of shape (1, 4000) for CNN input, or None if missing and allow_missing=True.
        label
    """
    def __init__(
        self,
        csv_path: str,
        label_name: List[str],
        spectra_dir: str,
        allow_missing: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.allow_missing = allow_missing

        # Track which mol_ids have available spectrum files
        available = {
            os.path.splitext(f)[0]
            for f in os.listdir(spectra_dir)
            if f.endswith(".npy")
        }

        if not allow_missing:
            # Original behavior: filter to only samples with spectrum data
            self.df = self.df[self.df["mol_id"].astype(str).isin(available)].reset_index(drop=True)
        else:
            # New behavior: keep all samples and track which have spectrum data
            self.available_ids = available

        self.labels      = self.df[label_name].values.astype(float)
        self.spectra_dir = spectra_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        mol_id  = row['mol_id']
        path    = os.path.join(self.spectra_dir, f"{mol_id}.npy")

        # Check if spectrum file exists when allow_missing is True
        if self.allow_missing and str(mol_id) not in self.available_ids:
            # Return None for missing spectrum data
            spec = None
        else:
            vec = np.load(path)
            spec = torch.from_numpy(vec)   # Convert the binary spectrum vector to torch tensor
            spec = spec.to(torch.float32)

        label = torch.from_numpy(self.labels[idx]).to(torch.float32)

        return spec, label
    
class GraphSMILESDataset(Dataset):
    """
    Multi-modal dataset combining graph and SMILES data.
    Args:
        g (GraphDataset), s (SMILESDataset)
    """
    def __init__(self, g, s):
        self.g = g
        self.s = s
    def __len__(self):
        return len(self.g)
    def __getitem__(s, i):
        g = s.g[i]
        ids, mask, label = s.s[i]
        return g, ids, mask, label

class GraphImageDataset(Dataset):
    """
    Multi-modal dataset combining graph and image data.
    Args:
        g (GraphDataset), i (ImageDataset)
    """
    def __init__(self, g, i):
        self.g = g
        self.i = i
    def __len__(self): return len(self.g)
    def __getitem__(self,i):
        g = self.g[i]
        img,y = self.i[i]
        return g, img, y
    

class GraphSpectrumDataset(Dataset):
    """
    Multi-modal dataset combining graph and spectra data.
    Args:
        g (GraphDataset), sp (SpectrumDataset)
    """
    def __init__(self, g, sp):
        self.g, self.sp = g, sp
    def __len__(self):
        return len(self.g)
    def __getitem__(self, idx):
        graph = self.g[idx]     
        ppm_list, y2 = self.sp[idx]  
        return graph, ppm_list, y2

class SMILESImageDataset(Dataset):
    """
    Multi-modal dataset combining SMILES and image data.
    Args:
        s (SMILESDataset), i (ImageDataset)
    """
    def __init__(self, s, i):
        self.s, self.i = s, i
    def __len__(self): return len(self.s)
    def __getitem__(self, idx):
        ids, mask, _ = self.s[idx]
        img, y = self.i[idx]
        return ids, mask, img, y

class SMILESSpectrumDataset(Dataset):
    """
    Multi-modal dataset combining SMILES and spectrum data.
    Args:
        s (SMILESDataset), sp (SpectrumDataset)
    """
    def __init__(self, s, sp):
        self.s, self.sp = s, sp
    def __len__(self): return len(self.s)
    def __getitem__(self, idx):
        ids, mask, _ = self.s[idx]
        ppm_list, y = self.sp[idx]
        return ids, mask, ppm_list, y

class SpectrumImageDataset(Dataset):
    """
    Multi-modal dataset combining spectrum and image data.
    Args:
        sp (SpectrumDataset), i (ImageDataset)
    """
    def __init__(self, sp, i):
        self.sp, self.i = sp, i
    def __len__(self): return len(self.sp)
    def __getitem__(self, idx):
        ppm_list, _ = self.sp[idx]
        img, y = self.i[idx]
        return ppm_list, img, y

class GraphSMILESImageDataset(Dataset):
    """
    Multi-modal dataset combining graph, image, and SMILES data.
    Args:
        g (GraphDataset), s (SMILESDataset), i (ImageDataset)
    """
    def __init__(self, g, s, i):
        self.g, self.s, self.i = g, s, i
    def __len__(self): return len(self.g)
    def __getitem__(s,i):
        g = s.g[i]
        img,y = s.i[i]
        ids, mask,_ = s.s[i]
        return g, ids, mask, img, y 
    
class MoltiToxDataset(Dataset):
    """
    Dataset used in the MoltiTox model for multimodal molecular property prediction.
    Combines molecular graph, 2D structure image, 13C NMR spectra, and SMILES sequence for each compound.
    Args:
        g (GraphDataset),  s (SMILESDataset), i (ImageDataset), sp (SpectrumDataset)
    """
    def __init__(self, g, s, i, sp): 
        self.g, self.s, self.i, self.sp = g, s, i, sp
    def __len__(self):
        return len(self.g)
    def __getitem__(self, i):
        g = self.g[i]
        img, y = self.i[i]
        ppm_list, _ = self.sp[i]
        ids, mask, _ = self.s[i]
        return g, ids, mask, img, ppm_list, y