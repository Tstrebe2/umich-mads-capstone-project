import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

world_size    = int(os.getenv("WORLD_SIZE"))
rank          = int(os.getenv("SLURM_PROCID"))
gpus_per_node = int(os.getenv("SLURM_GPUS_ON_NODE"))

dist.init_process_group("nccl", rank=rank, world_size=world_size)