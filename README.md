## How to run training using `SLURM`
Before running training please adjust `checkpoint_dir` variable.
- SA-MIRI's network:
```bash
sbatch job_cte_nostrategy.sh --architecture sa_miri --epochs 20 --batch_size 128
```
```bash
sbatch job_cte_mirroredstrategy.sh --architecture sa_miri --epochs 20 --batch_size 128
```
```bash
sbatch job_cte_multiworkermirroredstrategy.sh --architecture sa_miri --epochs 20 --batch_size 128
```

- MobileNet:
```bash
sbatch job_cte_nostrategy.sh --architecture mobilenet --epochs 5 --batch_size 128
```
```bash
sbatch job_cte_mirroredstrategy.sh --architecture mobilenet --epochs 5 --batch_size 128
```
```bash
sbatch job_cte_multiworkermirroredstrategy.sh --architecture mobilenet --epochs 5 --batch_size 128
```

- Resnet101:
```bash
sbatch job_cte_nostrategy.sh --architecture resnet101 --epochs 5 --batch_size 64
```
```bash
sbatch job_cte_mirroredstrategy.sh --architecture resnet101 --epochs 5 --batch_size 64
```
```bash
sbatch job_cte_multiworkermirroredstrategy.sh --architecture resnet101 --epochs 5 --batch_size 64
```

## Results
Evaluations made intra-node (using 2-4 GPUs) were using `NCCLAllReduce`, evaluations made inter-node (using 8+ GPUs) were using `RING` collective communication.
- SA-MIRI's network:
![SA-MIRI's network training speedup](images/mnist_samiri_speedup.png)
- MobileNet:
![MobileNet training speedup](images/cifar_mobilenet_speedup.png)
- Resnet101:
![Resnet101 training speedup](images/cifar_resnet101_speedup.png)

## Future work
It would be interesting to measure speedup for `MultiWorkerMirroredStrategy` with `NCCL` set as collective communication mechanism.