import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        print("Number of GPUs available:", num_gpus)
        for i in range(num_gpus):
            print("Device ID:", i, "Name:", torch.cuda.get_device_name(i))
    else:
        print("No GPUs available.")
else:
    print("CUDA is not available.")
