import argparse
import os
import Main

""" Experiment Setting """ 
# ARGUMENT
parser = argparse.ArgumentParser(description='Fatigue')
parser.add_argument('--data_root', default='DATASET_DIR/', help="name of the data folder")
parser.add_argument('--run_code_folder', default='')
parser.add_argument('--save_root', default='./MODEL_SAVE_DIR/', help="where to save the models and tensorboard records")
parser.add_argument('--result_dir', default="", help="save folder name")
parser.add_argument('--total_path', default="", help='total result path')
parser.add_argument('--cuda', type=bool, default=True, help='cuda')
parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
parser.add_argument('--device', default="", help='device')

parser.add_argument('--n_classes', type=int, default=0, help='num classes')
parser.add_argument('--n_channels', type=int, default=0, help='num channels')
parser.add_argument('--n_timewindow', type=int, default=0, help='timewindow')

## AUGMENTATION and ALIGNMENT
parser.add_argument('--res_layer', type=int, default=[], help='after which residual layer is MixStyle used')
parser.add_argument('--loss', default="AlignLoss", help='type of loss')
parser.add_argument('--aug_prob', default=0.5, help='probability of applying augmentation (80% augmentaion: 0.8, 20% augmenation: 0.2')
parser.add_argument('--align_weight', type=float, default=[], help="alignment loss weight")

parser.add_argument('--optimizer', default="Adam", help='optimizer')
parser.add_argument('--lr', type=float, default=0, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0, help='the amount of weight decay in optimizer') 
parser.add_argument('--scheduler', default="CosineAnnealingLR", help='scheduler')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size of each subject for training (default: 16)')
parser.add_argument('--valid-batch-size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for ONLINE testing (default: 1)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number worker') 

parser.add_argument('--subject_group', type=int, default=7, metavar='N', help='subject_group') 
parser.add_argument('--steps', type=int, default=0, help='Number of steps')
parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint every N steps')
parser.add_argument('--seed', type=int, default=2032, help='seed') 

parser.add_argument('--model_name', default='resnet8', help='trained model name')
parser.add_argument('--mode', default='train', help='train, infer')

parser.add_argument('--eval_metric', default=["acc", "bacc", "f1"], help='evaluation metric for model selection ["loss", "acc", "bacc", "f1"]')
parser.add_argument('--metric_dict', default={"loss":0,"acc":1, "bacc":2, "f1":3}, help='total evaluation metric')
parser.add_argument('--mix_type', default='', help='type of mixup: upmix, style')

parser.add_argument('--tensorboard', default=[], help='tensorboard writter')

args = parser.parse_args()
args=vars(args)

            
args["run_code_folder"]=os.path.realpath(__file__) # current folder name of running code
args["n_classes"]=2
args["n_channels"]=32
args["n_timewindow"]=600
args["lr"]=0.01
args["weight_decay"]=0
args['steps']=2200
args['optimizer']="Adam"

subjectList=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11'] 



model_name='resnet18'
args['loss']='AlignLoss'
args['mix_type'] = "style"   
args['align_weight']=0.5  
Main.main(subjectList, args, "resnet18") 

