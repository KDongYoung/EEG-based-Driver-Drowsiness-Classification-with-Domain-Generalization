import random
import numpy as np
import os
import pandas as pd
from datetime import datetime

from Data_Load.DataLoader import init_dataset
from Trainer import Trainer

from Networks.Model_ResNet1D18 import Resnet18

def Experiment(args, subject_id, subjectList):
   
    # make a directory to save results, models
    args['total_path'] = args['save_root'] + str(args['seed']) + "_" + str(args['steps']) + "_" + str(args['align_weight']) + "/" + args['result_dir']
    
    if not os.path.isdir(args['total_path']):
        os.makedirs(args['total_path'])
        os.makedirs(args['total_path'] + '/models/acc')
        os.makedirs(args['total_path'] + '/models/bacc')
        os.makedirs(args['total_path'] + '/models/f1')
        os.makedirs(args['total_path'] + '/models/loss')

    # save ARGUEMENT
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("\n"+str(args)+"\n\n")

    # connect GPU/CPU
    import torch.cuda
    args['cuda'] = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## fix seed
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args['cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

      
    if args['mode']=="train":
        valid_best = start_Training(args, subjectList, subject_id, seed) # Train
        args['mode']="infer"
        t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = start_Inference(args, subjectList, subject_id, seed) # Leave-one-subject-out 
        args['mode']="train"
      
    if args['mode']=="infer":
        t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = start_Inference(args, subjectList, subject_id, seed) # Leave-one-subject-out 
        valid_best = [0]*len(args['eval_metric'])
   
    return valid_best, t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost

def start_Training(args, subjectList, subject_id, seed):  # train
    num_domain=len(subjectList)-1
    
    # MODEL
    model = load_model(args, num_domain)
    if args['cuda']: model.cuda(device=args['device']) # connect DEVICE
    
    train_loaders, valid_loader, test_loader, test_set = init_dataset(args, subject_id, subjectList, seed) # load dataloader
    
    trainer = Trainer(args, subjectList, subjectList, subject_id, model)
    valid_best = trainer.training(train_loaders, valid_loader, test_loader) # train, valid
    
    return valid_best
    
def start_Inference(args, subjectList, subject_id, seed): # prediction  
    num_domain=len(subjectList)-1
    
    t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = [], [], [], [], [], [], [], []
    
    test_loader = init_dataset(args, subject_id, subjectList, seed) # load dataloader
    
    for metric in args['eval_metric']:
        # MODEL
        best_model = load_model(args, num_domain)    
        if args['cuda']: best_model.cuda(device=args['device']) # connect DEVICE
        
        best_trainer=Trainer(args, subjectList, subjectList, subject_id, best_model)
        
        loss, acc, bacc, f1, preci, rocauc, recall, cost = best_trainer.prediction(metric, test_loader)
        cost = np.mean(cost[1:]) # buffer 
        
        t_loss.append(loss)
        t_acc.append(acc)
        t_bacc.append(bacc)
        t_f1.append(f1)
        t_preci.append(preci)
        t_rocauc.append(rocauc)
        t_recall.append(recall)
        t_timecost.append(cost)
        
    return t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost
    
def load_model(args, num_domain): # load model
    if args['model_name']=="resnet18":
        model=Resnet18(args, num_domain)

    return model


'''
###########################################################################################
#  Main
###########################################################################################
'''    

def main(subjectList, args, model_name):

    args['model_name']=model_name
    
    exp_type=f"{model_name}_{args['mix_type']}{''.join([str(l) for l in args['res_layer']])}_{args['align_weight']}{args['loss']}"

    args['result_dir']=exp_type
    result_path = args['save_root'] + str(args['seed']) + "_" + str(args['steps']) + "_" + str(args['align_weight']) + "/Results"

    if not os.path.isdir(result_path):
        os.makedirs(result_path+"/loss")
        os.makedirs(result_path+"/acc")
        os.makedirs(result_path+"/bacc")
        os.makedirs(result_path+"/f1")

    for id in range(len(subjectList)):
        print(str(args['seed']) + " / " + str(args['align_weight']) + " / " + exp_type)   
        print("~"*25 + ' Valid Subject ' + subjectList[id] + " " + "~"*25)

        valid_best, loss, acc, bacc, f1, preci, auc, recall, time_cost = Experiment(args, before_sbj_num+id, subjectList)
        
        ### print performance
        if args["mode"]=="train":
            valid_best=[valid_best[args["metric_dict"][i]] for i in args["eval_metric"]]
      
        total_perf = np.array([[subjectList[id]]*len(args['eval_metric']), valid_best, loss, acc, bacc, f1, preci, recall, auc, time_cost])
        
        print("> "*20)
        for metric_i, metric_name in enumerate(args['eval_metric']):
            print(f"{metric_name} Valid: {valid_best[metric_i]:.2f}, TEST_SUBJECT: {subjectList[id]}, ACCURACY: {acc[metric_i]:.4f}%, PRECISION: {preci[metric_i]:.4f}%, RECALL: {recall[metric_i]:.4f}%")
            with open(result_path+"/"+metric_name+"/"+metric_name+"_"+str(args['seed'])+"_"+exp_type+'_Accuracy.txt', 'a') as f:
                f.write(f"valid_best: {valid_best[metric_i]:.2f}, {subjectList[id]}, acc: {acc[metric_i]:.4f}, bacc: {bacc[metric_i]:.4f}, f1: {f1[metric_i]:.4f}, precision: {preci[metric_i]:.4f}, recall: {recall[metric_i]:.4f}\n") # save test performance   
        
        
        df=pd.DataFrame(total_perf)
        for metric_i, metric_name in enumerate(args['eval_metric']):
            part=df.iloc[:, metric_i]
            df_part=part.to_frame()
        
            print(df_part)
            df_part.to_csv(result_path+"/"+metric_name+"/"+metric_name+"_"+str(args['seed'])+"_"+exp_type+".csv", mode="a", index=False)
            # "best value during validation", "loss", "accuracy", "balanced accuracy", "f1-score", "precision", "recall", "auroc", "time cost to predict once"
