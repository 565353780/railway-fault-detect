import torch
import numpy as np
import os

import cv2

from LapNet import LAPNet
from create_dataset import createDataset
from torch.nn import DataParallel
from collections import OrderedDict
from torch.nn.parameter import Parameter
import json
import base64
import numpy as np
import matplotlib.pyplot as plt

from flask import Flask, request, Response
app = Flask(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ModelName = "LapNet_chkpt_better_epoch3978_GPU0_line.pth"
# ModelName = "LapNet_chkpt_better_epoch1890_GPU0_chLi_line.pth"
DetectMode = "line"
Port = "9360"

# ModelName = "LapNet_chkpt_better_epoch1839_GPU0_point.pth"
# DetectMode = "point"
# Port = "9361"

class LapNet_Test:
    def __init__(self, model_name, detect_mode):
        # torch.cuda.set_device(args.gpu_idx)
        torch.cuda.set_device(0)

        # self.INPUT_CHANNELS = 3
        # self.OUTPUT_CHANNELS = 2
        # self.LEARNING_RATE = args.lr #1e-5
        # self.BATCH_SIZE = args.batch_size #20
        # self.NUM_EPOCHS = args.epoch #100
        # self.LOG_INTERVAL = 20
        # self.INS_CH = 32
        # self.SIZE = [args.img_size[0], args.img_size[1]] #[224, 224]
        # self.NUM_WORKERS = args.num_workers #20

        self.INPUT_CHANNELS = 3
        self.OUTPUT_CHANNELS = 2
        self.LEARNING_RATE = 3e-4
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10000000000000
        self.LOG_INTERVAL = 20
        self.INS_CH = 32
        self.SIZE = [1024,512]
        self.NUM_WORKERS = 32

        self.model_name = model_name
        self.detect_mode = detect_mode

        self.root_path = '../../../thirdparty/lapnet-gpu'

        self.model = LAPNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS,internal_ch = 8).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, betas=(0.9, 0.99), amsgrad=True)

        chkpt_filename = self.root_path + '/trained_model/'+ self.model_name

        if not os.path.exists(self.root_path + '/trained_model'):
            os.mkdir(self.root_path + '/trained_model')
        if os.path.isfile(chkpt_filename):
            checkpoint = torch.load(chkpt_filename)
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['net'])
            self.load_state_dict(self.model, self.state_dict(self.model))

    def state_dict(self, model, destination=None, prefix='', keep_vars=False):
        own_state = model.module if isinstance(model, torch.nn.DataParallel) \
            else model
        if destination is None:
            destination = OrderedDict()
        for name, param in own_state._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in own_state._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in own_state._modules.items():
            if module is not None:
                self.state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, model, state_dict, strict=True):
        own_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
            else model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                        'whose dimensions in the model are {} and '
                                        'whose dimensions in the checkpoint are {}.'
                                        .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                                .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

os.chdir("D:\chLi\Project\ABACI\RailwayFaultDetect\src\Python\lapnet")
lapnet_test = LapNet_Test(ModelName, DetectMode)
lapnet_test.model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data= request.get_data()
    data_json = json.loads(data)

    img_b64encode = bytes(data_json["Image"], encoding="utf-8")
    
    img_b64decode = base64.b64decode(img_b64encode)

    img_array = np.frombuffer(img_b64decode, np.uint8)
    image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

    train_dataset = createDataset("", size=lapnet_test.SIZE, image=image)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, pin_memory=True,
                                                shuffle=False, num_workers=0)
            
    img = list(enumerate(train_dataloader))[0][1]

    img_tensor = torch.tensor(img).cuda()

    sem_pred = lapnet_test.model(img_tensor)

    seg_map = torch.squeeze(sem_pred, 0).cpu().detach().numpy()

    seg_show = seg_map[1]

    _, seg_show2 = cv2.threshold(seg_show + 1, 0, 0, cv2.THRESH_TOZERO)
    seg_show2 = cv2.normalize(seg_show2, seg_show2, 0, 1, cv2.NORM_MINMAX)
    seg_show2 = cv2.convertScaleAbs(seg_show2, seg_show2, 255)
    result_img = cv2.applyColorMap(seg_show2, cv2.COLORMAP_MAGMA)

    output_img_array = cv2.imencode(".jpg", result_img)[1]

    output_img_b64encode = str(base64.b64encode(output_img_array))[2:-1]

    image_output_json = {}

    image_output_json["OutputImage"] = output_img_b64encode

    return image_output_json

class LapNet_Checker:
    def __init__(self, image_path, show_image):
        self.image_path = image_path
        self.show_image = show_image

        self.image_list = []
        self.similar_line_list = [[], [], [], [], []]

        self.load_image()

    def load_image(self):
        temp_image_list = os.listdir(self.image_path)

        for image_name in temp_image_list:
            if image_name[-4:] == ".jpg":
                if image_name.split(".")[0] + "_dif.jpg" in temp_image_list:
                    continue
                if image_name.split(".")[0] + ".png" in temp_image_list:
                    self.image_list.append(self.image_path + image_name[:-4])
    
    def predict(self, image_path):
        image = cv2.imread(image_path)

        train_dataset = createDataset("", size=lapnet_test.SIZE, image=image)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, pin_memory=True,
                                                    shuffle=False, num_workers=0)
                
        img = list(enumerate(train_dataloader))[0][1]

        img_tensor = torch.tensor(img).cuda()

        sem_pred = lapnet_test.model(img_tensor)

        seg_map = torch.squeeze(sem_pred, 0).cpu().detach().numpy()

        seg_show = seg_map[1]

        _, seg_show2 = cv2.threshold(seg_show + 1, 0, 0, cv2.THRESH_TOZERO)
        seg_show2 = cv2.normalize(seg_show2, seg_show2, 0, 1, cv2.NORM_MINMAX)
        seg_show2 = cv2.convertScaleAbs(seg_show2, seg_show2, 255)
        result_img = cv2.applyColorMap(seg_show2, cv2.COLORMAP_MAGMA)

        return result_img 
 
    def aHash(self, img):
        # 均值哈希算法
        # 缩放为8*8
        img_shape = img.shape
        # 转换为灰度图
        if len(img_shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # s为像素和初值为0，hash_str为hash值初值为''
        s = 0
        hash_str = ''
        # 遍历累加求像素和
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                s = s+gray[i, j]
        # 求平均灰度
        avg = s/ (img_shape[0] * img_shape[1])
        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                if gray[i, j] > avg:
                    hash_str = hash_str+'1'
                else:
                    hash_str = hash_str+'0'
        return hash_str
    
    
    def dHash(self, img):
        # 差值哈希算法
        # 缩放8*8
        img_shape = img.shape
        # 转换灰度图
        if len(img_shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        hash_str = ''
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(img_shape[0]):
            for j in range(img_shape[1] - 1):
                if gray[i, j] > gray[i, j+1]:
                    hash_str = hash_str+'1'
                else:
                    hash_str = hash_str+'0'
        return hash_str
    
    
    def pHash(self, img):
        # 感知哈希算法
        # 缩放32*32
        img_shape = img.shape   # , interpolation=cv2.INTER_CUBIC
    
        # 转换为灰度图
        if len(img_shape) != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 将灰度图转为浮点型，再进行dct变换
        dct = cv2.dct(np.float32(gray))
        # opencv实现的掩码操作
        dct_roi = dct[0:img_shape[0], 0:img_shape[1]]
    
        hash = []
        avreage = np.mean(dct_roi)
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash
    
    
    def calculate(self, image1, image2):
        # 灰度直方图算法
        # 计算单通道的直方图的相似值
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + \
                    (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
            else:
                degree = degree + 1
        degree = degree / len(hist1)
        return degree
    
    
    def classify_hist_with_split(self, image1, image2, size=(256, 256)):
        # RGB每个通道的直方图相似度
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        image1 = cv2.resize(image1, size)
        image2 = cv2.resize(image2, size)
        sub_image1 = cv2.split(image1)
        sub_image2 = cv2.split(image2)
        sub_data = 0
        div_num = 0
        for im1, im2 in zip(sub_image1, sub_image2):
            sub_data += self.calculate(im1, im2)
            div_num += 1
        sub_data = sub_data / div_num
        return sub_data
    
    
    def cmpHash(self, hash1, hash2):
        # Hash值对比
        # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
        # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
        # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        return n
    
    def runAllImageSimilaryFun(self, img1, img2):
        # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
        # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1

        img_size = img1.shape[0] * img1.shape[1]
    
        hash1 = self.aHash(img1)
        hash2 = self.aHash(img2)
        n1 = 1.0 - self.cmpHash(hash1, hash2) * 1.0 / img_size
        # print('均值哈希算法相似度aHash：', n1)
    
        hash1 = self.dHash(img1)
        hash2 = self.dHash(img2)
        n2 = 1.0 - self.cmpHash(hash1, hash2) * 1.0 / img_size
        # print('差值哈希算法相似度dHash：', n2)
    
        hash1 = self.pHash(img1)
        hash2 = self.pHash(img2)
        n3 = 1.0 - self.cmpHash(hash1, hash2) * 1.0 / img_size
        # print('感知哈希算法相似度pHash：', n3)
    
        n4 = self.classify_hist_with_split(img1, img2)[0]
        # print('三直方图算法相似度：', n4)
    
        n5 = self.calculate(img1, img2)[0]
        # print("单通道的直方图", n5)

        return n1, n2, n3, n4, n5
    
    def image_process(self, image):
        shape = image.shape

        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        output_image = cv2.threshold(output_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        return output_image
    
    def compare_image_and_result(self, image, result, image_index):
        difference = cv2.subtract(image, result)

        result_pixel_num = np.where(result > 200)[0].size
        difference_pixel_num = np.where(difference > 200)[0].size

        difference_scale = 1.0 * difference_pixel_num / result_pixel_num
        if result_pixel_num == 0:
            difference_scale = -1

        if self.show_image or difference_scale > 1 or True:
            # difference = cv2.subtract(image, result)
            # cv2.imshow("source_mask", image)
            # cv2.imshow("result_image", result)
            # cv2.imshow("difference", difference)
            difference_color_image = np.zeros((image.shape[0], image.shape[1], 3))
            image_color = [0, 0, 255]
            result_color = [255, 0, 0]
            common_color = [0, 255, 0]
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if image[i][j] > 200:
                        if result[i][j] > 200:
                            difference_color_image[i][j] = common_color
                        else:
                            difference_color_image[i][j] = image_color
                    elif result[i][j] > 200:
                            difference_color_image[i][j] = result_color
            save_image_path = self.image_list[image_index] + "_dif.jpg"
            cv2.imwrite(save_image_path, difference_color_image)
            # cv2.imshow("difference_color_image", difference_color_image)
            # cv2.waitKey(0)

        return difference_scale
        # return self.runAllImageSimilaryFun(image, result)

    def check_net_with_index(self, image_index):
        source_image_path = self.image_list[image_index] + ".jpg"

        source_mask = cv2.imread(self.image_list[image_index] + ".png")
        source_mask = cv2.resize(source_mask, (1024, 512))
        source_mask = self.image_process(source_mask)

        result_image = self.predict(source_image_path)
        result_image = self.image_process(result_image)

        n1 = self.compare_image_and_result(source_mask, result_image, image_index)
        # n1, n2, n3, n4, n5 = self.compare_image_and_result(source_mask, result_image)

        self.similar_line_list[0].append(n1)
        # self.similar_line_list[0].append(n1)
        # self.similar_line_list[1].append(n2)
        # self.similar_line_list[2].append(n3)
        # self.similar_line_list[3].append(n4)
        # self.similar_line_list[4].append(n5)

        if self.show_image:
            cv2.imshow("source_mask", source_mask)
            cv2.imshow("result_image", result_image)
            cv2.waitKey(0)

    def check_net(self):
        for i in range(len(self.image_list)):
            self.check_net_with_index(i)
            print("solved : ", i+1, "/", len(self.image_list))
    
    def show_similar_lines(self):
        similar_line_length = len(self.similar_line_list[0])

        x = range(similar_line_length)

        plt.plot(x, self.similar_line_list[0], ".", marker=".")
        # for i in range(5):
        #     plt.subplot(5, 1, i+1)
        #     plt.title("n" + str(i+1))
        #     plt.plot(x, self.similar_line_list[i], ".", marker=".")
        
        plt.show()

if __name__ == "__main__":
    lapnet_checker = LapNet_Checker("Z:/public/MaskStation/2c_allImage/MaskSpace/LapNet/train_line/", False)
    lapnet_checker.check_net()
    lapnet_checker.show_similar_lines()

    exit()

    app.run(host="0.0.0.0", port=Port,debug=True)
