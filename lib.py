'''
shared functions between scripts
'''
def apply_args(parser):
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-shuffle', '--shuffle', action='store_true', help='shuffle')
    parser.add_argument('-epochs', '--epochs', type=int, default=1, help='epochs')
    return parser
