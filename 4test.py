import wrapped_flappy_bird as game
import tensorflow as tf
import cv2
import numpy as np
from src.search_space.networks import *

ACTIONS = 2 

game_state = game.GameState()


#初始化状态并且预处理图片，把连续的四帧图像作为一个输入（State）
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
s_t, r_0, terminal, _ = game_state.frame_step(do_nothing)
s_t = cv2.cvtColor(cv2.resize(s_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, s_t = cv2.threshold(s_t,1,255,cv2.THRESH_BINARY)

model = Network(3, 2, 1, eval(str("Genotype(normal=[('none', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5], reduce=[('none', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])")))

while terminal !=True:
    a_t_to_game = np.zeros([ACTIONS])
    action_index = 0
    
    s_t = cv2.cvtColor(cv2.resize(s_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = torch.tensor(s_t, dtype=torch.float32)
    s_t = s_t.permute(2, 0, 1)
    s_t = s_t.unsqueeze(0)

    print(s_t.shape)

    readout_t = model(s_t)
    action_index = np.argmax(readout_t.detach().cpu().numpy())
    a_t_to_game[action_index] = 1

    s_t, r_t, terminal, score = game_state.frame_step(a_t_to_game)
    print("============== score ====================")
    print(score)