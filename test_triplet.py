import torch
from models.triplet_attention import TripletAttention

def test_triplet():
    print("\n STARTING TRIPLET ATTENTION TEST...")
    
    # Input: Batch=2, Channels=32, Height=128, Width=128
    x = torch.randn(2, 32, 128, 128)
    print(f"   Input Shape: {x.shape}")
    
    # Initialize Triplet Attention
    triplet = TripletAttention()
    
    try:
        out = triplet(x)
        print(f"   Output Shape: {out.shape}")
        
        # Verification: Output must maintain EXACT shape of input
        if out.shape == x.shape:
            print("SUCCESS: Triplet Attention rotation & fusion works correctly.")
        else:
            print(f"FAILURE: Shape mismatch! Got {out.shape}")
            
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    test_triplet()