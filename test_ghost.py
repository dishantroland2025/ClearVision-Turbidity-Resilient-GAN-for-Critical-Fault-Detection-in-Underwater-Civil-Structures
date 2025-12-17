import torch
from models.ghost_module import GhostModule

def test_ghost():
    print("\n STARTING GHOST MODULE TEST...")
    
    # Dummy input: Batch=1, Channels=32, Height=256, Width=256
    input_tensor = torch.randn(1, 32, 256, 256)
    print(f"   Input Shape: {input_tensor.shape}")

    # Initialize the module
    # Transform 32 channels -> 64 channels
    model = GhostModule(inp=32, oup=64, kernel_size=3, ratio=2)
    
    # Run the model
    try:
        output = model(input_tensor)
        print(f"   Output Shape: {output.shape}")
        
        if output.shape == (1, 64, 256, 256):
            print("SUCCESS: Output shape matches expected dimensions.")
        else:
            print("FAILURE: Output shape is wrong.")
            
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    test_ghost()