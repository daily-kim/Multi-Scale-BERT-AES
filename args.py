import torch, configargparse

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--efl_encode', action='store_true', help='is continue training')
    p.add('--r_dropout', help='r_dropout', type=float)
    p.add('--batch_size', help='batch_size', type=int)
    p.add('--bert_batch_size', help='bert_batch_size', type=int)
    p.add('--cuda', action='store_true', help='use gpu or not')
    p.add('--device')
    p.add('--model_directory', help='model_directory')
    p.add('--test_file', help='test data file')
    p.add('--data_dir', help='data directory to store asap experiment data')
    p.add('--data_sample_rate', help='data_sample_rate', type=float)
    p.add('--prompt', help='prompt')
    p.add('--fold', help='fold')
    p.add('--chunk_sizes', help='chunk_sizes', type=str)
    p.add('--result_file', help='pred result file path', type=str)

    p.add('--train_file', help='train data file')
    p.add('--valid_file', help='valid data file')
    p.add('--bert_init',action='store_true',help='bert_init')
    p.add('--num_epochs', help='num_epochs', type=int)
    p.add('--lr', help='learning_rate', type=float)

    p.add('--alpha', help='alpha', type=float)
    p.add('--beta', help='beta', type=float)
    p.add('--gamma', help='gamma', type=float)
    

    args = p.parse_args()
    args.train_file = "%s/%s_fold%s_train.txt" % ( args.data_dir,args.prompt, args.fold)
    args.valid_file = "%s/%s_fold%s_valid.txt" % ( args.data_dir,args.prompt, args.fold)
    args.test_file = "%s/%s_fold%s_test.txt" % (args.data_dir,args.prompt, args.fold )
    args.model_directory = "%s/%s_%s" % (args.model_directory, args.prompt, args.fold)

    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda'
    else:
        args.dev = 'cpu'
    return args

