from dataloaders.datasets import sevenscenes, cambridge, sevenscenes_line, cambridge_line
from torch.utils.data import DataLoader

def make_data_loader(cfg, val_only=False, num=-1, **kwargs):
    if cfg["dataset"] == "7scenes":
        if cfg["landmark"] == "point":
            if not val_only:
                train_set = sevenscenes.SevenScenesSegmentation(cfg, split="train")
            val_set = sevenscenes.SevenScenesSegmentation(cfg, num, split="test")
        elif cfg["landmark"] == "line":
            if not val_only:
                train_set = sevenscenes_line.SevenScenesSegmentation(cfg, split="train")
            val_set = sevenscenes_line.SevenScenesSegmentation(cfg, num, split="test")
        else:
            raise NotImplementedError
    elif cfg["dataset"] == "cambridge":
        if cfg["landmark"] == "point":
            if not val_only:
                train_set = cambridge.CambridgeSegmentation(cfg, split="train")
            val_set = cambridge.CambridgeSegmentation(cfg, num, split="test")
        elif cfg["landmark"] == "line":
            if not val_only:
                train_set = cambridge_line.CambridgeSegmentation(cfg, split="train")
            val_set = cambridge_line.CambridgeSegmentation(cfg, num, split="test")
        else:
            raise NotImplementedError        
    else:
        raise NotImplementedError
    
    if not val_only:
        train_loader = DataLoader(train_set, batch_size=cfg["train_batch_size"], shuffle=cfg["shuffle"], drop_last=True, **kwargs)
        test_loader = None
    val_loader = DataLoader(val_set, batch_size=cfg["val_batch_size"], shuffle=False, **kwargs)

    #return train_loader, val_loader, test_loader, train_set
    if not val_only:
        return train_loader, val_loader, test_loader, val_set
    else:
        return val_loader 

