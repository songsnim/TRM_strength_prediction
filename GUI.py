import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import captum
import shap

import torch
import torch.nn as nn
from tkinter import *

mean_dict={'b': 176.9625,
            'h': 242.3,
            'd': 207.9835,
            "d'": 25.825,
            'L': 2441.0625,
            'l': 2215.75,
            'a': 794.25,
            'fck': 30.697625,
            'As': 110.73037500000001,
            'fy': 357.99725,
            "As'": 230.72625,
            "fy'": 494.216,
            'C': 0.3625,
            'PBO': 0.1875,
            'G': 0.15,
            'CF': 0.075,
            'B': 0.0375,
            'ff': 3120.6375,
            'Af': 14.595875000000001,
            'layer': 2.125,
            'Swr': 9.932500000000001,
            'Swf': 13.569999999999999,
            'Anc': 0.325}

std_dict={'b': 96.34890291928602,
            'h': 81.40061424829668,
            'd': 72.65806763704909,
            "d'": 18.162994659471767,
            'L': 899.6255101950755,
            'l': 849.2920213330631,
            'a': 266.5298998236408,
            'fck': 8.828589814312078,
            'As': 107.1200526330592,
            'fy': 236.85287821754986,
            "As'": 194.7459760455592,
            "fy'": 93.60362070454326,
            'C': 0.4807221130757352,
            'PBO': 0.3903123748998999,
            'G': 0.3570714214271425,
            'CF': 0.26339134382131846,
            'B': 0.18998355191963331,
            'ff': 2076.1850245326764,
            'Af': 30.569788300450742,
            'layer': 2.0938899206978383,
            'Swr': 6.126331998022961,
            'Swf': 9.863523711128797,
            'Anc': 0.4683748498798799}


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
        
class mlp_model(nn.Module):
    def __init__(self):
        super(mlp_model, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(23,1024),
            nn.ELU(),
            nn.Linear(1024,512), 
            nn.ELU(),
            nn.Linear(512,512),  
            nn.ELU(), 
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,1))

    def forward(self,x):
        self.mlp.apply(weight_init_xavier_uniform)
        output = self.mlp(x)
        return output

def get_result():
    input_list = []
    for feature in feature_list:
        scaled_feature = (float(globals()[f"ent_{feature}"].get())-mean_dict[feature])/std_dict[feature]
        input_list.append(scaled_feature)
    
    output = MLP(input_list)
    result.configure(text=str(int(output)))

def MLP(input_list):
    
    mlp_input = torch.tensor(np.array(input_list), dtype=torch.float32)
    
    output = final_model(mlp_input)
    
    return(output)
    
    
if __name__ == '__main__':
    
    feature_list = ['b', 'h', 'd', "d'", 'L', 'l', 'a', 'fck', "As'", "fy", 'As', "fy'",
                'C', 'PBO', 'G', 'CF', 'B', 'ff', 'Af', 'layer', 'Swr', 'Swf','Anc']
    
    mlp = mlp_model().to("cpu")
    final_model = torch.load('best_model_GUI.pt', map_location='cpu')
    feat_width = 5
    
    # 프레임 생성
    root = Tk()
    root.title("TRM 항복강도 예측 프로그램")
    root.geometry("1035x400+100+100")
    
    lab_list = []
    ent_list = []
    
    for idx, feature in enumerate(feature_list):
        if idx <= len(feature_list)//2:
            globals()[f"lab_{feature}"] = Label(root,
                                                text=feature,
                                                width=feat_width,
                                                height=1,
                                                font=('Jetbrains Mono',16,'bold'),
                                                bg='#2F5597',
                                                fg='white'
                                                )
            globals()[f"lab_{feature}"].grid(row=0, column=idx, padx=5, pady=5)
            
            globals()[f"ent_{feature}"]=Entry(font=('맑은 고딕',16,'bold'),bg='white',width=feat_width)
            globals()[f"ent_{feature}"].grid(row=1, column=idx, padx=5, pady=5)
            
            # lab_list.append(globals()[f"lab_{feature}"])
            # ent_list.append(globals()[f"ent_{feature}"])
            
            
        elif idx > len(feature_list)//2:
            globals()[f"lab_{feature}"] = Label(root,
                                                text=feature,
                                                width=feat_width,
                                                height=1,
                                                font=('Jetbrains Mono',16,'bold'),
                                                bg='#2F5597',
                                                fg='white'
                                                )
            globals()[f"lab_{feature}"].grid(row=2, column=idx-len(feature_list)//2-1, padx=5, pady=5)
            globals()[f"ent_{feature}"]=Entry(font=('맑은 고딕',16,'bold'),bg='white',width=feat_width)
            globals()[f"ent_{feature}"].grid(row=3, column=idx-len(feature_list)//2-1, padx=5, pady=5)
            
            # lab_list.append(globals()[f"lab_{feature}"])
            # ent_list.append(globals()[f"ent_{feature}"])
    

    
    result_button = Button(root,
                      text='결과 계산',
                      font=('Jetbrains Mono',11,'bold'),
                      bg="red",fg='white',
                      width=feat_width,height=1,
                      command=get_result
                      )
    result_button.grid(row=5,column=idx-len(feature_list)//2-1,padx=5,pady=10,columnspan=4,sticky='we')

    result = Label(root,font=('맑은 고딕',16,'bold'),bg='yellow', fg="black",width=feat_width)
    result.grid(row=6, column=idx-len(feature_list)//2-1, 
                padx=5, pady=5, columnspan=2)

    root.mainloop()
