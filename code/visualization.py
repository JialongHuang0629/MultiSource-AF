import numpy as np
from tqdm import tqdm
from skimage import io

# 可视化 data 中的数据
def visualization(net, data, save_path, device, color_map, size):
    net.eval()
    h, w = size[:]
    pred = -np.ones((h, w))
    for hsi, sar, i, j in tqdm(data):
        hsi = hsi.to(device)
        sar = sar.to(device)
        output = net(hsi, sar)
        output = np.argmax(output.detach().cpu().numpy(), axis=1)
        idx = 0
        for x, y in zip(i, j):
            pred[x, y] = output[idx]
            idx += 1
    res = np.zeros((h, w, 3), dtype=np.uint8)
    pos = pred > -1
    for i in range(h):
        for j in range(w):
            if pos[i, j]:
                res[i, j] = color_map[int(pred[i, j])]
            else:
                res[i, j] = [0, 0, 0]
    io.imsave(save_path, res)

# 可视化 Berlin 数据集
def visBerlin(net, data, save_path, device):
    # Berlin color map
    berlin_color_map = [[47, 102, 57], [183, 40, 46], [178, 180, 182], [166, 199, 127],[153,78,47], [103, 127, 63], [200, 179, 204],
                          [137, 177,209]]
    # Berlin 尺寸
    berlin_size = [1723, 476]
    print("Berlin Start!")
    visualization(net, data, save_path, device, berlin_color_map, berlin_size)
    print("Visualization Success!")

# 可视化 Augsburg 数据集
def visAugsburg(net, data, save_path, device):
    # Augsburg color map
    augsburg_color_map = [[47, 102, 57], [183, 40, 46], [178, 180, 182], [166, 199, 127], [103, 127, 63], [200, 179, 204],
                          [137, 177,209]]
    # Augsburg 尺寸
    augsburg_size = [332, 485]
    print("Augsburg Start!")
    visualization(net, data, save_path, device, augsburg_color_map, augsburg_size)
    print("Visualization Success!")

# 可视化 Houston2018 数据集
def visHouston2018(net, data, save_path, device):
    # Houston2018 color map
    houston2018_color_map = [[50, 205, 51], [173, 255, 48], [0, 128, 129], [34, 139, 34], [46, 79, 78], [139, 69, 18], [0, 255, 255], [255, 255, 255], [211, 211, 211], [254, 0, 0], [169, 169, 169], [105, 105, 105], [139, 0, 1], [200, 100, 0], [254, 165, 0], [255, 255, 0], [218, 165, 33], [255, 0, 254], [0, 0, 254], [63, 224, 208]]
    # Houston2018 尺寸
    houston2018_size = [1202, 4768]
    print("Houston2018 Start!")
    visualization(net, data, save_path, device, houston2018_color_map, houston2018_size)
    print("Visualization Success!")

def getMyVisualization(datasetType, net, data, save_path, device):
    if(datasetType == 'berlin'):
        visBerlin(net, data, save_path, device)
    elif(datasetType == 'augsburg'):
        visAugsburg(net, data, save_path, device)
    elif(datasetType == 'Houston'):
        visHouston2018(net, data, save_path, device)