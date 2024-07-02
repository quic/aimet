##  Quantization-Aware Training Knowledge Distillation (QAT-KD)
### How to run
An example training pipeline of QAT-KD is shown in `finetune_llm_qat_kd.py`.
This script relies on [deepspeed](https://github.com/microsoft/DeepSpeed) for distributed training, for which you need to run each worker process with proper environment variables such as `$MASTER_ADDR`, `$MASTER_PORT`, `$WORLD_SIZE`, `$RANK`, `$LOCAL_RANK`, etc.
One fast and easy way to automate this procedures is using deepspeed launcher that comes with deepspeed PyPI package.

#### To run on single node
```bash
# Run finetune_llm_qat_kd.py on GPU 0~7 on localhost
deepspeed --master_port <port_number> --include "localhost:0,1,2,3,4,5,6,7" finetune_llm_qat_kd.py
```

#### To run on multiple nodes
```bash
# Run finetune_llm_qat_kd.py on GPU 0,1,2,3 on hostname_0 and GPU 0,1,2,3 on hostname_1.
deepspeed --hostfile hostfile.txt \
    --master_addr hostname_0 \
    --master_port <port_number> \
    --include "hostname_0:0,1,2,3@hostname_1:0,1,2,3" \
    finetune_llm_qat_kd.py

# NOTE: hostfile.txt is a text file that lists up all the nodes and GPUs which looks like below:
# --- hostfile.txt example ---
# hostname_0:0,1,2,3,4,5,6,7
# hostname_1:0,1,2,3,4,5,6,7
# hostname_2:0,1,2,3,4,5,6,7
# ----------------------------
# Note that deepspeed launcher will use all the resources enlisted in hostfile.txt if
# --include option is not specified.
```
