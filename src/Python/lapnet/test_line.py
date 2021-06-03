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

from flask import Flask, request, Response
app = Flask(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ModelName = "LapNet_chkpt_better_epoch3978_GPU0_line.pth"
# ModelName = "LapNet_chkpt_better_epoch1890_GPU0_chLi_line.pth"
DetectMode = "line"
Port = "9360"

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Port,debug=True)
