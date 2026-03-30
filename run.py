"""
函数主入口
"""
import argparse
import datetime
from utils.ch_train_competitive import ChRun_com, ChConfig_com
from utils.MCFM_train import EnConfig_com, EnRun_com


def main(args):
    if args.dataset != 'sims':
        EnRun_com(EnConfig_com(batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
                               text_model=args.text_model, audio_model=args.audio_model, dataset_name=args.dataset,
                               num_hidden_layers=args.num_hidden_layers))
    else:
        ChRun_com(ChConfig_com(batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
                               text_model=args.text_model, audio_model=args.audio_model, model=args.model,
                               tasks=args.tasks,
                               cme_version=args.cme_version, num_hidden_layers=args.num_hidden_layers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--dataset', type=str, default='mosei', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=5,
                        help='number of hidden layers for cross-modality encoder')
    args = parser.parse_args()

    if args.dataset == 'mosi' or args.dataset == 'mosei':
        args.text_model = r'E:\MPSFM\llms\roberta_base'
        args.audio_model = r'E:\MPSFM\llms\data2vec_audio_base'
    elif args.dataset == 'sims':
        args.text_model = r'D:\desktop\models--bert-base-chinese'
        args.audio_model = 'E:\MPSFM\llms\hubert'

    print('Training Argument')
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(datetime.datetime.now())
    main(args)
