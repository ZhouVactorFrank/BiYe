import os
import cv2
import numpy as np
import torch
from albumentations import Compose
import albumentations as A

# model_path = r"C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet_vgg19_noscSE\e50_model_0.pth"
# model_path = r"C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet++_resnet34_noscSE_trainBatch=8\e50_model_1.pth"
model_path = r"C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet++_vgg19_scSE_trainBatch=4\e50_model_2.pth"
# model_path = r"C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet++_resnet34_scSE_trainBatch=8\e50_model_2.pth"

# 定义模型和设备
model = torch.load(model_path, map_location="cpu")  # Unet++_vgg19_scSE_trainBatch=4 cuda显存不足 使用cpu跑预测图
# model = torch.load(model_path)  # 其余3个模型基于gpu（cuda）进行预测
model.eval()

# 定义测试数据的转换函数
test_transform = Compose([
    A.Normalize()
])

# 定义输入文件夹和输出文件夹
input_folder_A = r"C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\test\A"
input_folder_B = r"C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\test\B"
output_folder = r"C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\result_Unet++_vgg19_scSE"
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename_A in os.listdir(input_folder_A):
    filename_B = os.path.basename(filename_A)
    filepath_A = os.path.join(input_folder_A, filename_A)
    filepath_B = os.path.join(input_folder_B, filename_B)

    # 读取图像
    img_A = cv2.imread(filepath_A)
    img_B = cv2.imread(filepath_B)

    # 转换图像
    img_A = test_transform(image=img_A)['image']
    img_B = test_transform(image=img_B)['image']

    # 转换为 PyTorch 张量
    img_A = img_A.transpose(2, 0, 1)
    img_B = img_B.transpose(2, 0, 1)
    img_A = np.expand_dims(img_A, 0)
    img_B = np.expand_dims(img_B, 0)
    img_A = torch.Tensor(img_A)
    img_B = torch.Tensor(img_B)

    # 将图像输入模型
    with torch.no_grad():
        torch.cuda.empty_cache()
        img_A = img_A.to("cpu")
        # img_A = img_A.cuda()
        img_B = img_B.to("cpu")
        # img_B = img_B.cuda()
        pre = model(img_A, img_B).to("cpu")
        # pre = model(img_A, img_B)
        pre = torch.argmax(pre, dim=1).cpu().data.numpy()
        pre = pre * 255

    # 将预测结果写入到文件中
    output_filename = os.path.join(output_folder, filename_A)
    cv2.imwrite(output_filename, pre[0])
