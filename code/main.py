from train.model_train import main_train
from test.model_test import main_test
import parameter #存放参数



if __name__=='__main__':

    #初始化参数
    parameter._init()

    #程序入口
    main_train()
    main_test()



