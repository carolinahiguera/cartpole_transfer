import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# gravitys = [9.8, 9.8, 9.8, 9.8]
# masscarts = [1.0, 2.0, 3.0, 2.0]
# pole_lenghts = [0.5, 2.0, 2.0, 3.0]

# masscarts = [2.0, 3.0, 2.0, 3.0]
# pole_lenghts = [0.5, 0.5, 0.5, 0.5]
# # masspoles = [0.5, 0.5, 1.0, 1.0 ]
# masspoles = [2.0, 2.0, 3.0, 3.0 ]

# masscarts = [1.0]
# pole_lenghts = [0.5]

gravitys = [9.8*1.5, 9.8*2, 9.8*3, 9.8*4]


for task in range(0,1):    
    # p = f'masscart{masscarts[task]}_polelenght{pole_lenghts[task]}'
    #p = f'masscart{masscarts[task]}_masspoles{masspoles[task]}/'
    # p = f'gravity{gravitys[task]}/'
    p = f'T0-T1_2/'
    for i in range(1,11):
        print(f'Getting measures for {p} trial {i}')
        dpath = f'output/{p}/DQN_{i}'
        dname = os.listdir(dpath)
        dname.sort()
        ea = EventAccumulator(os.path.join(dpath, dname[0])).Reload()
        tags = ['episode_reward','input_info/rewards','loss/loss', 'loss/td_error']
        labels = {'episode_reward':'episode_reward', 
                'input_info/rewards':'input_info_rewards', 
                'loss/loss':'loss',
                'loss/td_error':'td_error'}

        for tag in tags:
            tag_values=[]
            steps=[]
            for event in ea.Scalars(tag):
                tag_values.append(event.value)        
                steps.append(event.step)
            data = np.column_stack((steps,tag_values))
            np.save(f'{dpath}/{labels[tag]}_{i}.npy', data)
    print('done')
