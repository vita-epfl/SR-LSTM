'''
Utils script
This script is modified from 'https://github.com/YuejiangLIU/social-lstm-pytorch' by Anirudh Vemula
Author: Pu Zhang
Date: 2019/7/1
'''
import torch
import gc
import os
import pickle
import numpy as np
import scipy.linalg as sl
import random

## TrajNet++
import trajnetplusplustools
from data_load_utils import prepare_data
from trajnet_loader import trajnet_loader

import copy
class DataLoader_bytrajec2():
    def __init__(self, args):

        self.args=args
        print("Dataset: ", args.dataset_name)
        
        # ======= Loading trajnet loaders =======
        # Loading train, val and test trajnet loaders 
        self.trainbatch, self.trainbatchnums = \
            self.load_trajnet_loader(self.args, 'train')
        print('Total number of training batches:', self.trainbatchnums)
        
        self.valbatch, self.valbatchnums = \
            self.load_trajnet_loader(self.args, 'val')
        print('Total number of validation batches:', self.valbatchnums)

        ###########################
        # TODO:
        # !!! Take care of the trajnet_loader for testing !!!
        # L177: pos_scene_obs_pred = pos_scene[:args.obs_len + args.pred_len]
        
        # !!! Uncomment it in the train script as well !!!

        # self.testbatch, self.testbatchnums = \
        #     self.load_trajnet_loader(self.args, 'test')
        # print('Total number of test batches:', self.testbatchnums)

        # Using the validation set as the test loader since the test_epoch 
        # function computes ADE/FDE as well
        self.testbatch, self.testbatchnums = \
            self.load_trajnet_loader(self.args, 'val')
        print('Total number of test batches:', self.testbatchnums)
        ###########################
        
        # =======================================


    def load_trajnet_loader(self, args, mode='train'):
        """
        The loader has to be of the same format as the "batch_data" lists in the 
        functions that act as batch getters (get_train_batch, ...).
        Afterwards the function rotate_shift_batch modifies it a bit and returns
        the final format of the data that's used during training.
        """
        # Construct the dataset
        loader, _, _ = prepare_data(
            'datasets/' + args.dataset_name, subset=f'/{mode}/', sample=args.sample
            )
        # Convert datasets to trajnet loaders
        traj_loader = trajnet_loader(
            loader, args, 
            drop_distant_ped=True, 
            fill_missing_obs=args.fill_missing_obs,
            keep_single_ped_scenes=args.keep_single_ped_scenes,
            test=(mode == 'test')
            ) 
        traj_loader = list(traj_loader)

        return traj_loader, len(traj_loader)

    
    def rotate_shift_batch(self,batch_data,ifrotate=True):
        ''' Random rotation and zero shifting. '''
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data
        # Rotate batch
        if ifrotate:
            th = random.random() * np.pi
            cur_ori = batch.copy()
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:,:, 1] * np.sin(th)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:,:, 1] * np.cos(th)
        # get shift value
        s = batch[self.args.obs_length - 1]

        shift_value = np.repeat(s.reshape((1, -1, 2)), self.args.seq_length, 0)

        batch_data = \
            batch, batch - shift_value, shift_value, \
            seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data,ifrotate=self.args.randomRotate)
        return batch_data,batch_id

    def get_val_batch(self, idx):
        batch_data, batch_id = self.valbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data,ifrotate=self.args.randomRotate)
        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data,ifrotate=False)
        return batch_data, batch_id


def getLossMask(outputs,node_first, seq_list,using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    '''
    seq_length = outputs.shape[0]
    node_pre=node_first
    lossmask=torch.zeros(seq_length,seq_list.shape[1])
    if using_cuda:
        lossmask=lossmask.cuda()
    for framenum in range(seq_length):
        lossmask[framenum]=seq_list[framenum]*node_pre
        if framenum>0:
            node_pre=seq_list[framenum-1]
    return lossmask,sum(sum(lossmask))

def L2forTest(outputs,targets,obs_length,lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error=torch.norm(outputs-targets,p=2,dim=2)
    #only calculate the pedestrian presents fully presented in the time window
    pedi_full=torch.sum(lossMask,dim=0)==seq_length
    error_full=error[obs_length-1:,pedi_full]
    error=torch.sum(error_full)
    error_cnt=error_full.numel()
    final_error=torch.sum(error_full[-1])
    final_error_cnt=error_full[-1].numel()

    return error.item(),error_cnt,final_error.item(),final_error_cnt,error_full
    
def L2forTest_nl(outputs,targets,obs_length,lossMask,seq_list,nl_thred):
    '''
    Evaluation including non-linear ade/fde.
    '''
    nl_list=torch.zeros(lossMask.shape).cuda()
    pednum=targets.shape[1]
    for ped in range(pednum):
        traj=targets[seq_list[:,ped]>0,ped]
        second=torch.zeros(traj.shape).cuda()
        first=traj[:-1]-traj[1:]
        second[1:-1]=first[:-1]-first[1:]
        tmp=abs(second)>nl_thred
        nl_list[seq_list[:,ped]>0,ped]=(torch.sum(tmp, 1)>0).float()
    seq_length = outputs.shape[0]

    error=torch.norm(outputs-targets,p=2,dim=2)
    error_nl =error*nl_list
    #only calculate the pedestrian presents fully presented in the time window
    pedi_full=torch.sum(lossMask,dim=0)==seq_length
    error_nl = error_nl[obs_length - 1:,pedi_full]
    error_full=error[obs_length-1:,pedi_full]
    error_sum=torch.sum(error_full)
    error_cnt=error_full.numel()
    final_error=torch.sum(error_full[-1])
    final_error_cnt=error_full[-1].numel()
    error_nl=error_nl[error_nl>0]
    return error_sum.item(),error_cnt,final_error.item(),final_error_cnt,torch.sum(error_nl).item(),error_nl.numel(),error_full

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod