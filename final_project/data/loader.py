from torch.utils.data import DataLoader

from data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    # get the actual dataset
    # read the comments for TrajectoryDataset class
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    # the dataset loader, which returns the data batch per batch, here we send the batch size as input
    # the collate function gets the data and returns the batch manually
    # we send the dset object here, this object is send to the collate_fn
    # where we curate the batch manually see seq_collate function in trajectories.py
    # one thing to note is that, the pytorch DataLoader class class the
    # __getitem__ function of our Dataset class i.e. TrajectoryDataset, batch_size number of times
    # hence 64 and then sends a list of the outputs to seq_collate function
    # read the details of seq_collate function
    loader = DataLoader(
        
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate)
    return dset, loader
