import wandb

wandb.init(project="wandb")

for my_metric in range(10):
    wandb.log({'my_metric': my_metric})