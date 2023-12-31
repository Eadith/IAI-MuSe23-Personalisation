import pathlib
from datetime import datetime
from typing import List, Dict

from glob import glob, escape

import pandas as pd
from dateutil import tz
import os
from argparse import ArgumentParser
from shutil import rmtree

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from config import AROUSAL, PERSONALISATION_DIMS, PERSONALISATION
from data_parser import load_personalisation_data
from dataset import MuSeDataset, custom_collate_fn
from eval import get_predictions
from main import get_eval_fn, get_loss_fn
from train import train_personalised_models
from utils import seed_worker, log_results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True, help='model id')
    parser.add_argument('--emo_dim', default=AROUSAL, choices=PERSONALISATION_DIMS,
                        help=f'Specify the emotion dimension, (default: {AROUSAL}).')
    parser.add_argument('--checkpoint_seed', required=False, help='Checkpoints to use, e.g. '
                                                                             '101 if for model that was trained with seed 101 '
                                                                             '(cf. output in the model directory). Not needed for --eval_personalised')
    parser.add_argument('--normalize', action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=20,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=10,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=3047,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--keep_checkpoints', action='store_true', help='Set this in order *not* to delete all the '
                                                                        'personalised checkpoints')
    parser.add_argument('--eval_personalised', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')

    args = parser.parse_args()
    if not args.eval_personalised:
        assert args.checkpoint_seed, f'Argument --checkpoint_seed is required.'
    args.timestamp = datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M-%S")
    args.run_name = f'{args.model_id}_{args.checkpoint_seed}_personalisation_{args.timestamp}'
    args.log_file_name = args.run_name
    args.paths = {'log': os.path.join(config.LOG_FOLDER, PERSONALISATION),
                  'data': os.path.join(config.DATA_FOLDER, PERSONALISATION),
                  'model': os.path.join(config.MODEL_FOLDER, PERSONALISATION, args.model_id,
                                        f'{args.checkpoint_seed}_personalised_{args.timestamp}' if not args.eval_personalised else args.eval_personalised)}
    if args.predict:
        if args.eval_personalised:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, PERSONALISATION, 'personalised', args.emo_dim,
                                                 args.model_id, args.eval_personalised)
        else:
            args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, PERSONALISATION, 'personalised', args.emo_dim,
                                                 args.model_id, args.run_name)
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    args.paths.update({'features': config.PATH_TO_FEATURES[PERSONALISATION],
                       'labels': config.PATH_TO_LABELS[PERSONALISATION],
                       'partition': config.PARTITION_FILES[PERSONALISATION]})
    args.model_file = os.path.join(config.MODEL_FOLDER, PERSONALISATION, args.model_id,
                                    f'model_{args.checkpoint_seed}.pth')
    # determine feature from filename
    args.feature = args.model_id.split("_")[2][1:-1]
    return args

def get_stats(arr):
    return {'mean':np.mean(arr), 'std':np.std(arr), 'min':np.min(arr), 'max':np.max(arr)}


def eval_personalised(personalised_cps:Dict[str, str], id2data_loaders:Dict[str, Dict[str, DataLoader]], use_gpu=False):
    """
    :param personalised_cps dictionary mapping subject ids to corresponding model files
    :param id2data_loaders dictionary mapping subject ids to their data
    :return for development and test set: predictions, score, dataframe with predictions and gold standard
    """
    all_dev_labels = []
    all_dev_preds = []
    all_dev_ids = []
    all_test_labels = []
    all_test_preds = []
    all_test_ids = []
    for subject_id, model_file in personalised_cps.items():
        model = torch.load(model_file, map_location=config.device)
        subj_dev_labels, subj_dev_preds = get_predictions(model=model, task=PERSONALISATION,
                                                          data_loader=id2data_loaders[subject_id]['devel'],
                                                          use_gpu=use_gpu)
        all_dev_labels.append(subj_dev_labels)
        all_dev_preds.append(subj_dev_preds)
        all_dev_ids.extend([subject_id]*subj_dev_labels.shape[0])

        subj_test_labels, subj_test_preds = get_predictions(model=model, task=PERSONALISATION,
                                                            data_loader=id2data_loaders[subject_id]['test'],
                                                            use_gpu=use_gpu)
        all_test_labels.append(subj_test_labels)
        all_test_preds.append(subj_test_preds)
        all_test_ids.extend([subject_id]*subj_test_labels.shape[0])
    all_dev_labels = np.concatenate(all_dev_labels)
    all_dev_preds = np.concatenate(all_dev_preds)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_preds = np.concatenate(all_test_preds)

    eval_fn, _ = get_eval_fn(PERSONALISATION)
    val_score = eval_fn(all_dev_preds, all_dev_labels)
    test_score = eval_fn(all_test_preds, all_test_labels)

    dev_df = pd.DataFrame({'meta_subj_id':all_dev_ids, 'pred':all_dev_preds, 'label_gs':all_dev_labels})
    test_df = pd.DataFrame({'meta_subj_id': all_test_ids, 'pred': all_test_preds, 'label_gs': all_test_labels})
    return (all_dev_preds, val_score, dev_df), (all_test_preds, test_score, test_df)


def create_data_loaders(data, test_ids):
    """
    :param data list of data for every subject, where each entry is a dict with keys 'train', 'devel', 'test'
    :param test_ids list of test subject ids
    """
    data_loaders = []
    for subj_data in data:
        data_loader = {}
        for partition in subj_data.keys():
            set = MuSeDataset(subj_data, partition)
            batch_size = args.batch_size if partition == 'train' else 1
            shuffle = True if partition == 'train' else False  # shuffle only for train partition
            data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle,
                                                                 num_workers=0,
                                                                 worker_init_fn=seed_worker,
                                                                 collate_fn=custom_collate_fn)
        data_loaders.append(data_loader)
    id2data_loaders = {i: d for i, d in zip(test_ids, data_loaders)}
    return data_loaders, id2data_loaders


def eval_trained_checkpoints(paths, feature, emo_dim, normalize, win_len, hop_len, cp_dir, use_gpu):
    """
    Evaluate a set of checkpoints as given in paths
    """
    data, test_ids = load_personalisation_data(paths, feature, emo_dim, normalize=normalize, win_len=win_len,
                                               hop_len=hop_len, save=True,
                                               segment_train=True)
    data_loaders, id2data_loaders = create_data_loaders(data, test_ids)
    # load personalised cps
    checkpoints = sorted([cp for cp in glob(os.path.join(escape(cp_dir), 'model_*.pth')) if 'initial' not in os.path.basename(cp)])
    initial_cp = os.path.join(cp_dir, 'model_initial.pth')
    initial_model = torch.load(initial_cp, map_location=config.device)
    initial_model.eval()
    personalised_cps = {os.path.splitext(os.path.basename(cp))[0].split("_")[1]:cp for cp in checkpoints}
    return eval_personalised(personalised_cps=personalised_cps, id2data_loaders=id2data_loaders,
                                                    use_gpu=use_gpu)


def personalise(model, feature, emo_dim, temp_dir, paths, normalize, win_len, hop_len, epochs, lr, use_gpu, loss_fn,
                eval_fn, eval_metric_str, early_stopping_patience, reduce_lr_patience, seeds, regularization=0.0):
    """
    Main part of the 2nd step. Take a general model and personalise it for every test subject
    """
    data, test_ids = load_personalisation_data(paths, feature, emo_dim, normalize=normalize, win_len=win_len, hop_len=hop_len, save=True,
                              segment_train=True)

    data_loaders, id2data_loaders = create_data_loaders(data, test_ids)

    # subject id to personalised model cp
    personalised_cps = train_personalised_models(model=model, temp_dir=temp_dir, data_loaders=data_loaders, subject_ids=test_ids, epochs=epochs,
                              lr=lr, use_gpu=use_gpu, loss_fn=loss_fn, eval_fn=eval_fn,
                                eval_metric_str=eval_metric_str, early_stopping_patience=early_stopping_patience,
                              reduce_lr_patience=reduce_lr_patience, regularization = regularization, seeds=seeds)

    v,t = eval_personalised(personalised_cps=personalised_cps, id2data_loaders=id2data_loaders,
                                                    use_gpu=use_gpu)
    val_score = v[1]
    test_score = t[1]
    return val_score, test_score


def log_personalisation_results(csv_path, params, val_score, test_score, metric_name,
                exclude_keys=['result_csv', 'cache', 'save', 'save_path', 'predict', 'eval_model', 'log_file_name']):
    """
    Logs result of a run into a csv
    :param csv_path: path to the desired csv. Appends, if csv exists, else creates it anew
    :param params: configuration of the run (parsed cli arguments)
    :param val_results: array of validation metric results
    :param test_results: array of test metric results
    :param best_idx: index of the chosen result
    :param model_files: list of saved model files
    :param metric_name: name of the used metric
    :param exclude_keys: keys in params not to consider for logging
    :return: None
    """
    dct = {k:[v] for k,v in vars(params).items() if not k in exclude_keys}
    dct.update({f'val_{metric_name}': val_score})
    dct.update({f'test_{metric_name}':test_score})
    df = pd.DataFrame(dct)

    # make sure the directory exists
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    # write back
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df])
    df.to_csv(csv_path, index=False)


def random_init(model:torch.nn.Module):
    for param in model.parameters():
        torch.nn.init.normal(param)
    return model


if __name__ == '__main__':
    args = parse_args()


    if not args.eval_personalised:
        model = torch.load(args.model_file, map_location=config.device)

        pers_dir = args.paths['model']

        eval_fn, eval_metric_str = get_eval_fn(PERSONALISATION)
        loss_fn, loss_fn_str = get_loss_fn(PERSONALISATION)
        seeds = list(range(args.seed, args.seed + args.n_seeds))

        val_score, test_score = personalise(model=model, feature=args.feature, emo_dim=args.emo_dim, temp_dir=pers_dir, paths=args.paths,
                    normalize=args.normalize, win_len=args.win_len, hop_len=args.hop_len, epochs=args.epochs, lr=args.lr,
                    use_gpu=args.use_gpu, loss_fn=loss_fn,
                    eval_fn=eval_fn, eval_metric_str=eval_metric_str, early_stopping_patience=args.early_stopping_patience,
                    reduce_lr_patience=args.reduce_lr_patience, regularization=args.regularization, seeds=seeds)
        print('Finished personalisation. Results:')
        print(f'[Val]: {val_score:7.4f}')
        print(f'[Test]: {test_score:7.4f}')
        if args.result_csv:
            log_personalisation_results(args.result_csv, params=args, metric_name=eval_metric_str, val_score=val_score,
                                        test_score=test_score)
        elif args.predict:
            _,t = eval_trained_checkpoints(
            paths=args.paths, feature=args.feature, emo_dim=args.emo_dim, normalize=args.normalize,
            win_len=args.win_len, hop_len=args.hop_len, cp_dir=args.paths['model'], use_gpu=args.use_gpu)
            _,_,test_df = t
            test_predict_csv = os.path.join(args.paths['predict'], 'predictions_test.csv')
            test_df.to_csv(test_predict_csv, index=False)
            print(f'Find test predictions in {args.paths["predict"]}')
        if not args.keep_checkpoints:
            rmtree(pers_dir)

    else:
        d,t = eval_trained_checkpoints(
            paths=args.paths, feature=args.feature, emo_dim=args.emo_dim, normalize=args.normalize,
            win_len=args.win_len, hop_len=args.hop_len, cp_dir=args.paths['model'], use_gpu=args.use_gpu)
        _, dev_score, dev_df = d
        _, test_score, test_df = t
        print(f'[Val]: {dev_score:7.4f}')
        print(f'[Test]: {test_score:7.4f}')
        if args.predict:
            dev_predict_csv = os.path.join(args.paths['predict'], 'predictions_devel.csv')
            dev_df.to_csv(dev_predict_csv, index=False)
            test_predict_csv = os.path.join(args.paths['predict'], 'predictions_test.csv')
            test_df.to_csv(test_predict_csv, index=False)
            print(f'Find predictions in {args.paths["predict"]}')


