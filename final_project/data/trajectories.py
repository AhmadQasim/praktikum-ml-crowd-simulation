import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

"""
The trajectories.py file is a slightly modified version of the
original trajectories.py file from GitHub.
https://github.com/agrimgupta92/sgan
"""


# this function creates a batch from the list of TrajectoryDataset outputs
# the length of the list is batch_size i.e. 64
def seq_collate(data):
    # upzip the TrajectoryDataset object
    # remember TrajectoryDataset returns
    # in order obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    # get a list of lengths of each sequence
    _len = [len(seq) for seq in obs_seq_list]
    # create a cumsum again
    cum_start_idx = [0] + np.cumsum(_len).tolist()

    # again, find the start and end of each sequence
    # we did this in TrajectoryDataset as well but we didn't send it in the __get_item__ output
    # so we do it here
    # also before it was for the whole dataset and now its only for this current batch
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # lstm requires the input format to be: [seq_len, number of all pedestrians in the batch, input_size]

    # we concatenate all the obs i.e. X sequence data and change the dimensions to match the required lstm
    # we will do this permute for all the data

    # shape: [obs_len, number of all pedestrians in the batch, input_size]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)

    # shape: [pred_len, number of all pedestrians in the batch, input_size]
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    # so our batch contains the sequences data all concatenated together
    # and it contains the start and end of each sequence
    # we use this start and end later on to find individual sequences within each batch
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


# read data rows with the delimiter
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


# checks if the trajectory is linear or not, returns 1 for non-linear and 0 for linear
def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        # array to save for each sequence, the number of pedestrians which are in that sequence
        # so we keep track of that
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            # read data with delimiter tab
            data = read_file(path, delim)
            print(all_files)
            # find the number of frames by applying unique function to first column (with frame IDs)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            # read all the data related to each frame i.e. read all columns from data where frame_id == frame iteration
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            # find the number of sequences
            # a sequence is an overlapping set of seq_len frames e.g. if seq_len = 20
            # first sequence: frames [0, 20]
            # second sequence: frames [1, 21]
            # and so on.. from this set obs_len are used as X and remaining pred_len as Y (ground truth)
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            # loop over all sequences
            for idx in range(0, num_sequences * self.skip + 1, skip):
                # get the current sequence data from frame data
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                # number of pedestrians in the current sequence by finding unique of the first column (pedestrians id)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                # initialize array for saving the relative positions of pedestrians in this sequence
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                # initialize array for saving the absolute positions of pedestrians in this sequence
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # loss mask, it will be initialized to 1 in the start
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                # loop over each pedestrian id in the sequence
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # get the data for current pedestrian from the curr_seq_data array
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    # round off the 4 decimal places
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # see if we are at the end of the frames data i.e. the length of remaining data is less then seq_len
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    # if its less the seq_len i.e. 20 then skip this data we are at the end
                    if pad_end - pad_front != self.seq_len:
                        continue
                    # get only the x, y coordinates of the current pedestrian data
                    # skip the frames_id and pedestrian id (:2) and transpose it
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # initialize the pedestrian wise relative pedestrian coordinates array
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # to get the relative coordinates subtract each subsequent position from last position
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    # set the curr_seq and curr_seq_rel arrays that we created before
                    # with the correct data for the pedestrian that we just extracted
                    # note that the shape of curr_seq is [pedestrians num, coordinates, sequence length]
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # check if the trajectory is linear or non-linear
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    # set the loss to 1
                    curr_loss_mask[_idx, pad_front:pad_end] = 1

                    # increment the peds considered in this sequence
                    # i think this variable simply keeps track if we skipped some pedestrian because the sequence
                    # length was less then seq_len see the continue statement above
                    num_peds_considered += 1

                # if the number of peds in the end in the frame are less then min pedestrians variable then
                # we will skip collecting data from that sequence
                if num_peds_considered > min_ped:
                    # increment the total non_linear peds variable
                    non_linear_ped += _non_linear_ped
                    # append to the list num_peds_in_seq, so we keep track of how many pedestrians are in which
                    # sequence
                    num_peds_in_seq.append(num_peds_considered)
                    # append to the loss mask list and other self explanatory lists
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        # total number of sequences that we collected the data for, remember we might have skipped some sequences
        # so num_sequences doesn't have the right number
        self.num_seq = len(seq_list)
        # all the concatenations of all the sequences data
        # concatenate all sequences data into one
        # remember the seq_list is a list of all sequences data of shape [pedestrians num, coordinates, sequence length]
        # so in the end the shape of seq_list after concatenation is:
        # [all pedestrians in all different sequences, coordinates, sequence length]
        # really important point, and a really shitty way to parse the data imo from the authors
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # numpy to torch tensors
        # get the observation data trajectories i.e. X from the data that we collected above
        # same for pred_traj, obs_traj_rel, pred_traj_rel, loss_mask and non_linear_ped
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        # create a cumsum of the number of pedestrians in each sequence
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # a list of tuples. Each tuple is the start and end of a sequence's pedestrian data
        # i think we have to keep track of this because concatenated all the sequences
        # but because the seq_list and other data is of shape
        # [all pedestrians in all different sequences, coordinates, sequence length]
        # so we need to keep track where one sequence ends and where the next starts
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    # return one item from the dataset, this function is called by pytorch dataset class, which is the super
    # class in this case
    def __getitem__(self, index):
        # pytorch sends the index as argument
        start, end = self.seq_start_end[index]


        # get the start and finish of the current sequence from the array that we saved before
        # remember [all pedestrians in all different sequences, coordinates, sequence length]
        # so we need to return the right sequence here
        # we return in order obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
