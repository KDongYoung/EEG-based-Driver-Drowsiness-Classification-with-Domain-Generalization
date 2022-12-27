from torch.utils.data import Dataset
import numpy as np
from scipy import io
import numpy as np

'''
Make a dataset
except kss score 7 (non-sleep: 1~6, sleep: 8~9)
'''
class DriverDrowsiness():
    def __init__(self,root_path, SUBJECT_LIST):
        if root_path is None:
            raise ValueError('Data directory not specified!')

        self.datasets=[]
        self.subjectList=SUBJECT_LIST
        self.sleep_num=0
        self.non_sleep_num=0
       
        for idx, SBJ_NAME in enumerate(self.subjectList):
            ORI_DATA=io.loadmat(root_path+SBJ_NAME+'.mat') 

            # 차원 축소
            self.x=np.squeeze(ORI_DATA["epoch"]["x"])
            self.x=self.x.flatten()[0]

            self.tr_df=np.transpose(self.x,(2,1,0)) # (n_segment, channel, time) shape
    
            #####
            wo7_idx=[] 
            for i in range(self.tr_df.shape[0]):
                kss=list(set(self.tr_df[i][-1].astype(int)))[0]
                
                if kss<7:
                    self.tr_df[i][-1]=0
                    wo7_idx.append(i)
                    self.non_sleep_num+=1
                elif kss>7:
                    self.tr_df[i][-1]=1
                    wo7_idx.append(i)
                    self.sleep_num+=1

            self.datasets.append(LG_EEGDataset(self.tr_df[wo7_idx], idx)) 
        print("alert : sleep =",self.non_sleep_num,":",self.sleep_num)

    def __getitem__(self, index):
        return self.datasets[index] # one eubject each

    def __len__(self):
        return len(self.datasets)
    
"""
Make a EEG dataset
X: EEG data
Y: KSS score
"""
class LG_EEGDataset(Dataset):
    def __init__(self, dataset, subj_id):
        self.dataset = dataset
        self.len = len(dataset)
        self.subj_id = subj_id
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.dataset[idx][0:32,0:600].astype('float32')  # for only eeg
        y = self.dataset[idx][-1,0].astype('int64')

        X=np.expand_dims(X,axis=0) # (1, channel, time) batch shape
                
        return X, y, self.subj_id

