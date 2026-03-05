def _init():
    global parameter
    parameter = {
        # net
        'model_name':'ours',
        'channels': 30,#30
        'windowSize': 10,
        'out_features': 8,
        # train
        'datasetType':'Berlin',
        'lr': 0.005,
        'epoch_nums': 100,
        'batch_size': 4,
        'num_workers': 0,
        'random_seed': 6,
        'visualization': True,
        'model_savepath': 'model/ours/Berlin_model',
        'log_path': 'log/ours/Berlin_log.txt',
        'report_path': 'report/ours/Berlin_report.txt',
        'image_path': 'pic/ours/Berlin.png'
    }

def set_value(key, value):
    parameter[key] = value

def get_value(key):
    try:
        return parameter[key]
    except:
        print('Read'+key+'failed\r\n')

def get_taskInfo():
    return '-----------------------TaskInfo----------------------- \n lr:\t{} \n epoch_nums:\t{} \n batch_size:\t{} \n window_size:\t{} \n------------------------------------------------------'.format(parameter['lr'], parameter['epoch_nums'], parameter['batch_size'], parameter['windowSize'])


