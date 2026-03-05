from core.utils.dataset import *
from report import *
from visualization import *
import parameter

parameter._init()

def main_test():
    device = torch.device("cuda:0")
    datasetType = parameter.get_value('datasetType')
    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    visualization = parameter.get_value('visualization')
    model_savepath = parameter.get_value('model_savepath')
    report_path = parameter.get_value('report_path')
    image_path = parameter.get_value('image_path')
    net = torch.load(model_savepath)
    train_loader, test_loader, trntst_loader, all_loader = fetchData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    getMyReport(datasetType, net, test_loader, report_path, device)
    if visualization:
        getMyVisualization(datasetType, net,trntst_loader, image_path, device)