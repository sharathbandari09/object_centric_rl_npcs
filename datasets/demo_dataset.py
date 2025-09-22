"""
Dataset class for loading KeyDoor demonstration data
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os

class DemoDataset(Dataset):
    """Dataset for loading KeyDoor demonstration data"""
    
    def __init__(self, data_type='raw', data_dir=None):
        """
        Initialize dataset
        
        Args:
            data_type: 'raw' or 'entity' 
            data_dir: Directory containing dataset files
        """
        self.data_type = data_type
        
        if data_dir is None:
            # Get the datasets directory relative to this file
            current_dir = Path(__file__).parent
            data_dir = current_dir
        
        # Load appropriate dataset
        if data_type == 'raw':
            dataset_file = data_dir / 'raw_keydoor_dataset_new.pkl'
        else:
            dataset_file = data_dir / 'entity_keydoor_dataset_new.pkl'
        
        try:
            with open(dataset_file, 'rb') as f:
                data = pickle.load(f)
            
            self.observations = data['observations']
            self.actions = data['actions']
            
            print(f"✅ Loaded {data_type} dataset: {len(self.observations)} samples")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        if self.data_type == 'raw':
            # Convert grid to tensor
            grid = torch.FloatTensor(obs['grid']).unsqueeze(0)  # Add channel dim
            return grid, action
        else:
            # Entity-based dataset
            entities = torch.FloatTensor(obs['entities'])
            entity_mask = torch.BoolTensor(obs['entity_mask'])
            return entities, entity_mask, action
    
    def get_info(self):
        """Get dataset information"""
        sample_obs = self.observations[0]
        
        info = {
            'total_samples': len(self.observations),
            'data_type': self.data_type,
            'action_distribution': np.bincount(self.actions, minlength=6).tolist()
        }
        
        if self.data_type == 'raw':
            info['grid_shape'] = sample_obs['grid'].shape
        else:
            info['entity_features'] = sample_obs['entities'].shape[1]
            info['max_entities'] = sample_obs['entities'].shape[0]
            
        return info

# Utility functions
def load_raw_dataset():
    """Load raw pixel dataset"""
    return DemoDataset(data_type='raw')

def load_entity_dataset():
    """Load entity-based dataset"""
    return DemoDataset(data_type='entity')

if __name__ == "__main__":
    # Test dataset loading
    print("🧪 Testing Dataset Loading")
    print("=" * 40)
    
    try:
        raw_dataset = load_raw_dataset()
        print(f"Raw dataset info: {raw_dataset.get_info()}")
        
        entity_dataset = load_entity_dataset()  
        print(f"Entity dataset info: {entity_dataset.get_info()}")
        
        print("✅ All datasets loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
