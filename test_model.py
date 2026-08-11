import torch
from src.models.dr_model import ICCIT_DR_Net
import sys

def run_tests():
    print("Running Model Architecture Smoke Tests...")
    dummy_input = torch.randn(2, 3, 384, 384)
    
    ablations = [
        ('baseline', False, False),
        ('msda_only', True, False),
        ('hff_only', False, True),
        ('proposed', True, True)
    ]
    
    for name, use_msda, use_hff in ablations:
        try:
            model = ICCIT_DR_Net(use_msda=use_msda, use_hff=use_hff, num_classes=5)
            out = model(dummy_input)
            assert out.shape == (2, 5), f"Shape mismatch for {name}: expected (2, 5), got {out.shape}"
            print(f"[OK] {name.upper()}: Output shape {out.shape}")
        except Exception as e:
            print(f"[ERROR] {name.upper()} failed: {e}")
            sys.exit(1)
            
    print("All architecture tests passed successfully!")

if __name__ == "__main__":
    run_tests()
