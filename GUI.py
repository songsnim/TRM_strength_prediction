import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from tkinter import *
from config import *

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
    final_model = torch.load('pretrained_models/best_model_GUI.pt', map_location='cpu')
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
