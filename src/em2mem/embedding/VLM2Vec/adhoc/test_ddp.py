import os
import torch
import torch.distributed as dist
import socket


def main():
    print(f"[Rank {os.environ.get('RANK')}] Hostname: {socket.gethostname()} | Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[rank {rank}] hostname: {os.uname().nodename}, MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"Starting rank {rank}, local rank {local_rank}, world size {world_size}")

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    print(f"Hello from rank {rank} out of {world_size}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()