import matplotlib.pyplot as plt
import re
import os

root_path = r'C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output'
with open(os.path.join(root_path, 'Unet_vgg19_noscSE\epoch_metrics_0.txt'), 'r') as f1, \
        open(os.path.join(root_path, 'Unet++_resnet34_noscSE_trainBatch=8\epoch_metrics_1.txt'), 'r') as f2, \
        open(os.path.join(root_path, 'Unet++_resnet34_scSE_trainBatch=8\epoch_metrics_2.txt'), 'r') as f3, \
        open(os.path.join(root_path, 'Unet++_vgg19_scSE_trainBatch=4\epoch_metrics_2.txt'), 'r') as f4:
    data1 = f1.read()
    data2 = f2.read()
    data3 = f3.read()
    data4 = f4.read()

pattern = r'Train Metrics: ({[^}]+})\s*Valid Metrics: ({[^}]+})'
matches1 = re.findall(pattern, data1)
matches2 = re.findall(pattern, data2)
matches3 = re.findall(pattern, data3)
matches4 = re.findall(pattern, data4)

loss1 = []
loss2 = []
loss3 = []
loss4 = []
for match in matches1:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    loss1.append(train_metrics['cross_entropy_loss'])
    # focal_loss1.append(valid_metrics['FocalLoss'])
for match in matches2:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    loss2.append(train_metrics['cross_entropy_loss'])
    # focal_loss2.append(valid_metrics['FocalLoss'])
for match in matches3:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    loss3.append(train_metrics['cross_entropy_loss'])
    # focal_loss3.append(valid_metrics['FocalLoss'])
for match in matches4:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    loss4.append(train_metrics['cross_entropy_loss'])
  
f1_score1_t = []
f1_score2_t = []
f1_score3_t = []
f1_score4_t = []
for match in matches1:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    f1_score1_t.append(train_metrics['fscore'])

for match in matches2:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    f1_score2_t.append(train_metrics['fscore'])

for match in matches3:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    f1_score3_t.append(train_metrics['fscore'])

for match in matches4:
    train_metrics, valid_metrics = match
    train_metrics = eval(train_metrics)
    valid_metrics = eval(valid_metrics)
    f1_score4_t.append(train_metrics['fscore'])

plt.plot(f1_score1_t, label='F1_UNet-V19', color='b', linestyle='-')
plt.plot(loss1, label='Loss_UNet-V19', color='b', linestyle='--')
plt.plot(f1_score2_t, label='F1_UNet++-R34', color='g', linestyle='-')
plt.plot(loss2, label='Loss_UNet++-R34', color='g', linestyle='--')
plt.plot(f1_score3_t, label='F1_UNet++-R34-scSE', color='r', linestyle='-')
plt.plot(loss3, label='Loss_UNet++-R34-scSE', color='r', linestyle='--')
plt.plot(f1_score4_t, label='F1_UNet++-V19-scSE', color='y', linestyle='-')
plt.plot(loss4, label='Loss_UNet++-V19-scSE', color='y', linestyle='--')

plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('$\mathit{value}$', fontsize=12)# 斜体
plt.ylabel('value', fontsize=12)
plt.legend(loc=(0.57, 0.2), fontsize=8, prop={'size': 8})  # 设置图例的起始位置
plt.xticks(range(0, 51, 5))
plt.grid(True)
plt.savefig(r'C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\result_0\loss_f1_score_loc=(0.57, 0.2).png', dpi=500) # 设置较高的dpi
plt.show()
