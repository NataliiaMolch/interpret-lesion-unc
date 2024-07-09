import torch


def get_scheduler(scheduler_name, learning_rate, optimizer, best_min_max, step_size, gamma, learning_rate_min):
    if scheduler_name == 'none':
        scheduler = None
    elif scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size,
                                                               eta_min=learning_rate_min)
    elif scheduler_name == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate_min,
                                                      max_lr=10 * learning_rate, step_size_up=step_size,
                                                      mode="triangular2", cycle_momentum=False)
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=best_min_max, factor=gamma,
                                                               patience=step_size,
                                                               min_lr=learning_rate_min, verbose=False)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    return scheduler