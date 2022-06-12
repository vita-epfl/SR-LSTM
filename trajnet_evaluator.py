import os
import argparse
import pickle
from platform import processor

from joblib import Parallel, delayed
import scipy
scipy.seterr('ignore')
import torch
from tqdm import tqdm
import trajnetplusplustools
import numpy as np

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import \
    load_test_datasets, preprocess_test, write_predictions

from trajnet_loader import trajnet_loader

# SR-LSTM
import ast
from yaml import CLoader as Loader, CDumper as Dumper
from Processor import *


def predict_scene(processor, batch_idx, args):
    # Put the model in eval mode
    processor.net.eval()

    # Get the corresponding batch
    batch, _ = processor.dataloader.get_test_batch(batch_idx)
    batch = [torch.Tensor(tensor).cuda() for tensor in batch]
    assert len(batch) == 7

    batch_abs, batch_norm, shift_value, \
    seq_list, nei_list, nei_num, batch_pednum = \
        batch
    inputs_fw = \
        batch_abs[:-1], batch_norm[:-1], shift_value[:-1], \
        seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum[:-1]

    # Get the predictions and save them
    multimodal_outputs = {}
    for num_p in range(args.modes):

        #############################
        # TODO:
        #   - when inputs_fw are passed, the output is of shape [19, ..., 2]
        #   - check what is the output shape in STGAT
        #   - check whether it would work if we pass inputs
        # Conclusion:
        #   - it should be fine just to keep the last pred_len frames

        outputs_infer, _, _, _ = \
            processor.net.forward(inputs_fw, iftest=True)
        
        outputs_infer = outputs_infer.detach().cpu().numpy()

        output_primary = outputs_infer[-args.pred_len:, 0]
        output_neighs = outputs_infer[-args.pred_len:, 1:]
        multimodal_outputs[num_p] = [output_primary, output_neighs]
        #############################

    # return multimodal_outputs



def load_predictor(args):
    processor = Processor(args)
    return processor


def get_predictions(args):
    """
    Get model predictions for each test scene and write the predictions 
    in appropriate folders.
    """
    # List of .json file inside the args.path 
    # (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) \
        if not f.startswith('.') and f.endswith('.ndjson')
        ])

    # Extract Model names from arguments and create its own folder 
    # in 'test_pred' for storing predictions
    # WARNING: If Model predictions already exist from previous run, 
    # this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print(f'Predictions corresponding to {model_name} already exist.')
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        processor = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = \
                load_test_datasets(dataset, goal_flag, args)

            # Convert it to a trajnet loader
            processor.create_dataloader_for_evaluator(scenes, zero_pad=True)

            # Get all predictions in parallel. Faster!
            pred_list = Parallel(n_jobs=args.n_jobs)(
                delayed(predict_scene)(processor, batch_idx, args)
                for batch_idx in tqdm(range(processor.dataloader.testbatchnums))
                )

            ###############
            # TODO:
            # # Write all predictions
            # write_predictions(pred_list, scenes, model_name, dataset_name, args)
            ###############


def get_parser():
    parser = argparse.ArgumentParser()

    # === Trajnet++ ===
    parser.add_argument("--trajnet_evaluator", default=1, type=int)
    parser.add_argument("--log_dir", default="./")
    parser.add_argument('--write_only', action='store_true')
    parser.add_argument("--dataset_name", default="colfree_trajdata", type=str)
    parser.add_argument("--delim", default="\t")
    parser.add_argument("--loader_num_workers", default=4, type=int)
    parser.add_argument("--obs_len", default=8, type=int)
    parser.add_argument("--pred_len", default=12, type=int)
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--fill_missing_obs", default=0, type=int)
    # In order to keep the collision test
    parser.add_argument("--keep_single_ped_scenes", default=1, type=int)
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=72, help="Random seed.")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--disable-collision', action='store_true')
    parser.add_argument('--labels', required=False, nargs='+')
    parser.add_argument('--normalize_scene', action='store_true')
    parser.add_argument('--modes', default=1, type=int)
    parser.add_argument("--n_jobs", default=8, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    # =================

    # ==== SR-LSTM ====
    # Always set to True for the evaluator
    parser.add_argument('--load_model', default=1,type=int)
    parser.add_argument('--using_cuda',default=True,type=ast.literal_eval) 
    # You may change these arguments (model selection and dirs)
    parser.add_argument('--test_set',default=0,type=int)
    parser.add_argument('--gpu', default=0,type=int,help='gpu id')
    parser.add_argument('--base_dir',default='.')
    parser.add_argument('--save_base_dir',default='./savedata/')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--train_model', default='srlstm',help='Your model name')
    parser.add_argument('--pretrain_model', default='')
    parser.add_argument('--pretrain_load', default=0,type=int)
    parser.add_argument('--model', default='models.SR_LSTM')

    parser.add_argument('--dataset',default='colfree_trajdata')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')

    parser.add_argument('--ifvalid',default=True,type=ast.literal_eval)
    parser.add_argument('--val_fraction',default=0.2,type=float)

    # Model parameters

    # LSTM
    parser.add_argument('--output_size',default=2,type=int)
    parser.add_argument('--input_embed_size',default=32,type=int)
    parser.add_argument('--rnn_size',default=64,type=int)
    parser.add_argument('--hidden_dot_size',default=32,type=int)
    parser.add_argument('--ifdropout',default=True,type=ast.literal_eval)
    parser.add_argument('--dropratio',default=0.1,type=float)
    parser.add_argument('--std_in',default=0.2,type=float)
    parser.add_argument('--std_out',default=0.1,type=float)

    # States Refinement
    parser.add_argument('--ifbias_gate',default=True,type=ast.literal_eval)
    parser.add_argument('--WAr_ac',default='')
    parser.add_argument('--ifbias_WAr',default=False,type=ast.literal_eval)
    parser.add_argument('--input_size',default=2,type=int)

    parser.add_argument('--rela_embed_size', default=32,type=int)
    parser.add_argument('--rela_hidden_size', default=16,type=int)
    parser.add_argument('--rela_layers', default=1,type=int)
    parser.add_argument('--rela_input', default=2,type=int)
    parser.add_argument('--rela_drop', default=0.1,type=float)
    parser.add_argument('--rela_ac', default='relu')
    parser.add_argument('--ifbias_rel',default=True,type=ast.literal_eval)

    parser.add_argument('--nei_hidden_size', default=64,type=int)
    parser.add_argument('--nei_layers', default=1,type=int)
    parser.add_argument('--nei_drop', default=0,type=int)
    parser.add_argument('--nei_ac', default='')

    parser.add_argument('--ifbias_nei',default=False,type=ast.literal_eval)
    parser.add_argument('--mp_ac', default='')
    parser.add_argument('--nei_std',default=0.01,type=float)
    parser.add_argument('--rela_std',default=0.3,type=float)
    parser.add_argument('--WAq_std',default=0.05,type=float)
    parser.add_argument('--passing_time',default=2,type=int)

    # Social LSTM
    parser.add_argument('--grid_size',default=4,type=int)
    parser.add_argument('--nei_thred_slstm',default=2,type=int)

    # Perprocess
    parser.add_argument('--seq_length',default=20,type=int)
    parser.add_argument('--obs_length',default=8,type=int)
    parser.add_argument('--pred_length',default=12,type=int)
    parser.add_argument('--batch_around_ped',default=128,type=int)
    parser.add_argument('--val_batch_size',default=8,type=int)
    parser.add_argument('--test_batch_size',default=4,type=int)
    parser.add_argument('--show_step',default=40,type=int)
    parser.add_argument('--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument('--ifdebug',default=False,type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False,type=ast.literal_eval)
    parser.add_argument('--randomRotate',default=True,type=ast.literal_eval)
    parser.add_argument('--neighbor_thred',default=10,type=int)
    parser.add_argument('--clip',default=1,type=int)
    parser.add_argument('--start_test',default=0,type=int)
    parser.add_argument('--learning_rate',default=0.0015,type=float)
    # =================

    return parser

def load_arg(p):
    # save arg
    if  os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=Loader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s=1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False

def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    # === Trajnet++ ===
    # Overwriting overlapping arguments with different names
    p.obs_length = p.obs_len
    p.pred_length = p.pred_len
    p.seq_length = p.obs_len + p.pred_len
    p.dataset = p.dataset_name
    # =================

    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config= p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)
    args = load_arg(p)
    torch.cuda.set_device(args.gpu)

    # Load the processor - he'll handle model predictions
    args.load_model = 1

    # Is this needed?
    args.checkpoint = os.path.join(
        args.save_dir, args.train_model, f"{args.train_model}_best.tar"
        )
    args.path = os.path.join('datasets', args.dataset_name, 'test_pred/')
    args.output = [args.checkpoint]

    # Adding arguments with names that fit the evaluator module
    # in order to keep it unchanged
    args.obs_length = args.obs_len
    args.pred_length = args.pred_len

    # Writes to Test_pred
    # Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)

