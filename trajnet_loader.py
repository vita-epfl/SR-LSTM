import numpy as np
import torch

import trajnetplusplustools


def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]


def get_limits_of_missing_intervals(finite_frame_inds, obs_len):
    """
    Given a SORTED array of indices of finite frames per pedestrian, get the 
    indices which represent limits of NaN (missing) intervals in the array.
    Example (for one pedestrian):
        array = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18]
        obs_len = 18

        ==>> result = [0, 3, 5, 8, 10, 13, 15, 18]
    The resulting array is an array with an even number of elements,
    because it represents pairs of start-end indices (i.e. limits) for 
    intervals that should be padded. 
        ==>> intervals to be padded later: [0, 3], [5, 8], [10, 13], [15, 18]
    """
    # Adding start and end indices
    if 0 not in finite_frame_inds:
        finite_frame_inds = np.insert(finite_frame_inds, 0, -1) 
    if obs_len not in finite_frame_inds:
        finite_frame_inds = \
            np.insert(finite_frame_inds, len(finite_frame_inds), obs_len)

    # Keeping only starts and ends of continuous intervals
    limits, interval_len = [], 1
    for i in range(1, len(finite_frame_inds)):
        # If this element isn't the immediate successor of the previous
        if finite_frame_inds[i] > finite_frame_inds[i - 1] + 1:
            if interval_len:
                # Add the end of the previous interval
                if finite_frame_inds[i - 1] == -1:
                    limits.append(0)
                else:
                    limits.append(finite_frame_inds[i - 1])
                # Add the start of the new interval
                limits.append(finite_frame_inds[i])
                # If this is a lone finite element, add the next interval
                if interval_len == 1 and i != len(finite_frame_inds) - 1 \
                    and finite_frame_inds[i + 1] > finite_frame_inds[i] + 1:
                    limits.append(finite_frame_inds[i])
                    limits.append(finite_frame_inds[i + 1])
            interval_len = 0
        else:
            interval_len += 1
            
    return limits


def fill_missing_observations(pos_scene_raw, obs_len, test):
    """
    Performs the following:
        - discards pedestrians that are completely absent in 0 -> obs_len
        - discards pedestrians that have any NaNs after obs_len
        - In 0 -> obs_len:
            - finds FIRST non-NaN and fill the entries to its LEFT with it
            - finds LAST non-NaN and fill the entries to its RIGHT with it
    """

    # Discarding pedestrians that are completely absent in 0 -> obs_len
    peds_are_present_in_obs = \
        np.isfinite(pos_scene_raw).all(axis=2)[:obs_len, :].any(axis=0)
    pos_scene = pos_scene_raw[:, peds_are_present_in_obs, :]

    if not test:
        # Discarding pedestrians that have NaNs after obs_len
        peds_are_absent_after_obs = \
            np.isfinite(pos_scene).all(axis=2)[obs_len:, :].all(axis=0)
        pos_scene = pos_scene[:, peds_are_absent_after_obs, :]

    # Finding indices of finite frames per pedestrian
    finite_frame_inds, finite_ped_inds = \
        np.where(np.isfinite(pos_scene[:obs_len]).all(axis=2))
    finite_frame_inds, finite_ped_inds = \
        finite_frame_inds[np.argsort(finite_ped_inds)], np.sort(finite_ped_inds)

    finite_frame_inds_per_ped = np.split(
        finite_frame_inds, np.unique(finite_ped_inds, return_index=True)[1]
        )[1:]
    finite_frame_inds_per_ped = \
        [np.sort(frames) for frames in finite_frame_inds_per_ped]

    # Filling missing frames
    for ped_ind in range(len(finite_frame_inds_per_ped)):
        curr_finite_frame_inds = finite_frame_inds_per_ped[ped_ind]

        # limits_of_cont_ints: [start_1, end_1, start_2, end_2, ... ]
        limits_of_missing_ints = \
            get_limits_of_missing_intervals(curr_finite_frame_inds, obs_len)
        assert len(limits_of_missing_ints) % 2 == 0
            
        i = 0
        while i < len(limits_of_missing_ints):
            start_ind, end_ind = \
                limits_of_missing_ints[i], limits_of_missing_ints[i + 1]
            # If it's the beginning (i.e. first element is NaN):
            #   - pad with the right limit, else use left
            #   - include start_ind, else exclude it
            if start_ind == 0 and not np.isfinite(pos_scene[0, ped_ind]).all():
                padding_ind = end_ind 
                start_ind = start_ind 
            else:
                padding_ind = start_ind
                start_ind = start_ind + 1

            pos_scene[start_ind:end_ind, ped_ind] = pos_scene[padding_ind, ped_ind]
            i += 2

    return pos_scene


def trajnet_loader(
    data_loader, 
    args, 
    drop_distant_ped=False, 
    test=False, 
    keep_single_ped_scenes=False,
    fill_missing_obs=False,
    zero_pad_pos_scene=False,
    ):
    """
    The SR-LSTM expects the return values to have the following format:
    Denoted as: {their_variable_name}: {our_variable_name} \n {shape}
        - batch: pos_scenes
            [num_frames, num_peds_batch, 2]
        - seq_list: peds_are_present
            [num_frames, num_peds_batch]
        - nei_list: adj_matrices_block_diag
            [num_frames, num_peds_batch, num_peds_batch]
        - nei_num: num_neighbors_per_frame
            [num_frames, num_peds]
        - batch_pednum: num_peds_per_batch
            [batch_size]
    """
    # Specific for SR-LSTM
    num_frames = args.seq_length
    pos_scenes, peds_are_present, adj_matrices, seq_start_end = [], [], [], []

    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        if test:
            paths = pre_process_test(paths, args.obs_len)
        
        ## Get new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        if drop_distant_ped:
            pos_scene = drop_distant(pos_scene)

        if fill_missing_obs:
            pos_scene = fill_missing_observations(pos_scene, args.obs_len, test)
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
        else:
            # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
            pos_scene = pos_scene[:, full_traj]
        
        if zero_pad_pos_scene:
            pos_scene_original = pos_scene.copy()
            num_frames_original = pos_scene_original.shape[0]
            pos_scene = np.zeros((num_frames, pos_scene_original.shape[1], 2))
            pos_scene[:num_frames_original, :, :] = pos_scene_original

        # === From now on it's specific for SR-LSTM ===
        if sum(full_traj) > 1 or keep_single_ped_scenes:
            pos_scene_obs_pred = pos_scene[:args.obs_len + args.pred_len]
            pos_scene_obs_pred = torch.Tensor(pos_scene_obs_pred)

            pos_scenes.append(torch.Tensor(pos_scene_obs_pred))
            peds_are_in_scene = \
                torch.isfinite(pos_scene_obs_pred).all(axis=2).type(torch.int8)
            peds_are_present.append(peds_are_in_scene)
            
            # Compute the adjacency matrices for the current scene
            adj_matrices_scene = []
            for t in range(num_frames):
                peds_are_in_frame = peds_are_in_scene[t, :].reshape(-1, 1)
                adj_matrix_frame = peds_are_in_frame @ peds_are_in_frame.T
                adj_matrices_scene.append(adj_matrix_frame)
            adj_matrices_scene = torch.stack(adj_matrices_scene)
            adj_matrices.append(adj_matrices_scene)

            # Get Seq Delimiter 
            seq_start_end.append(pos_scene.shape[1])

            num_batches += 1

        if num_batches % args.batch_size != 0 and (batch_idx + 1) != len(data_loader):
            continue
        
        if len(pos_scenes):
            pos_scenes = torch.cat(pos_scenes, dim=1)
            peds_are_present = torch.cat(peds_are_present, dim=1)

            # Create block-diagonal adjacency matrices
            #   - currently they are [20, p1, p1], [20, p2, p2], ...
            #   - should be [20, p1+p2+..., p1+p2+...]
            adj_matrices_block_diag_per_frame = []
            for t in range(num_frames):
                adj_matrices_frame_t = []
                for adj_matrix_scene in adj_matrices:
                    adj_matrices_frame_t.append(adj_matrix_scene[t, ...])

                # Appending a matrix of shape [p1+p2+..., p1+p2+...]
                # that corresponds to the t-th frame
                adj_matrices_block_diag_per_frame.append(
                    torch.block_diag(*adj_matrices_frame_t)
                    )

            # Stack them up all together => [20, p1+p2+..., p1+p2+...]
            adj_matrices_block_diag = torch.stack(adj_matrices_block_diag_per_frame)

            # Count the neighbors of each pedestrian in each frame
            num_neighbors_per_frame = torch.sum(adj_matrices_block_diag, axis=2)

            # Compute start-end sequences for pedestrians in the batch
            seq_start_end = [0] + seq_start_end
            seq_start_end = torch.LongTensor(np.array(seq_start_end).cumsum())
            seq_start_end = torch.stack((seq_start_end[:-1], seq_start_end[1:]), dim=1)

            # This tensor is of the following format:
            #   [[0 2], [3 5], [6 11], ...]; shape = [batch_size, 2]
            # and in SR-LSTM it's expected to be:
            #   [3, 3, 6, ...]; shape = [batch_size]
            num_peds_per_batch = seq_start_end[:, 1] - seq_start_end[:, 0] + 1

            batch_to_yield = (
                pos_scenes.numpy(), 
                peds_are_present.numpy(), 
                adj_matrices_block_diag.numpy(), 
                num_neighbors_per_frame.numpy(), 
                num_peds_per_batch.numpy()
                )

            # SR-LSTM also requires a 'batch_id' as a return value, 
            # but since it seems unimportant, we pass an empty string
            yield batch_to_yield, ''

            pos_scenes, peds_are_present = [], []
            adj_matrices, seq_start_end = [], []
        # =============================================
