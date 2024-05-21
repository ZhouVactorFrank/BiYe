mport json
import os

from torch.cuda.amp import autocast as autocast, GradScaler # 半精度训练 torch原生支持的amp

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# import os
# # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
# # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16'

# model = cdp.Unet(         # baseline Unet_vgg16
model = cdp.UnetPlusPlus(
    encoder_name="vgg19",   # Epoch: 34
                            # train: 100%|██████████| 112/112 [01:24<00:00,  1.32it/s, cross_entropy_loss - 0.3212, fscore - 0.7721, precision - 0.8093, recall - 0.8304, iou_score - 0.6934, accuracy - 0.9923]
                            # valid: 100%|██████████| 64/64 [00:31<00:00,  2.06it/s, cross_entropy_loss - 0.3216, fscore - 0.8172, precision - 0.8569, recall - 0.8285, iou_score - 0.7442, accuracy - 0.9918]
                            # max_score 0.8171805434548681
                            # Model saved!
    # encoder_name="resnet34",   # 自行设置编码器
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your datasets)
    siam_encoder=True,  # whether to use a siamese encoder
    fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
    decoder_attention_type="scse",
    activation="sigmoid", # 二分类使用sigmoid激活函数
).to(DEVICE)

train_dataset = LEVIR_CD_Dataset(r'C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\train',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir=r'C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\train\label',
                                 debug=False)

valid_dataset = LEVIR_CD_Dataset(r'C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\val',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir=r'C:\hangpian\DL_datasate\Building change detection dataset\LEVIR-CD\val\label',
                                 debug=False,
                                 test_mode=True)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0) # 剩下3个模型用的batch_size=8
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

loss = cdp.utils.losses.CrossEntropyLoss() # 交叉熵损失函数

metrics = [
    cdp.utils.metrics.Fscore(activation="argmax2d"),
    cdp.utils.metrics.Precision(activation="argmax2d"),
    cdp.utils.metrics.Recall(activation="argmax2d"),
    cdp.utils.metrics.IoU(activation="argmax2d"),  
    cdp.utils.metrics.Accuracy(activation="argmax2d"),  # Accuracy代表准确率(OA)
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),  # Unet++_vgg19_sese设置batch=4时为了recall不为1 设置初始学习率0.0001
                                                 # 其他3种情况都在batch=8 lr=0.001条件下进行
])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # SGD或许能减少显存占用

# 创建学习率衰减策略
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler_steplr = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = cdp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = cdp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 60 epochs

max_score = 0
MAX_EPOCH = 50

train_metrics_history = {
    'cross_entropy_loss': [],
    'fscore': [],
    'precision': [],
    'recall': [],
    'accuracy': [],
    'iou_score': []
}

valid_metrics_history = {
    'cross_entropy_loss': [],
    'fscore': [],
    'precision': [],
    'recall': [],
    'accuracy': [],
    'iou_score': []
}


def fill_missing_values(metrics_history, metric_name):
    for i in range(len(metrics_history[metric_name])):
        if metrics_history[metric_name][i] is None:
            j = i - 1
            while j >= 0:
                if metrics_history[metric_name][j] is not None:
                    metrics_history[metric_name][i] = metrics_history[metric_name][j]
                    break
                j -= 1


initial_metrics = {
    'cross_entropy_loss': 0.0,
    'fscore': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'accuracy': 0.0,
    'iou_score': 0.0
}

for key in initial_metrics.keys():
    train_metrics_history[key].append(initial_metrics[key])
    valid_metrics_history[key].append(initial_metrics[key])

# 设置保存路径
output_dir = r'C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet++_vgg19_scSE_trainBatch=4'
# output_dir = r'C:\Users\Administrator\Desktop\BiYeLunWenCode\change_detection.pytorch\LEVIR_output\Unet++_resnet34_noscSE_trainBatch=8'
os.makedirs(output_dir, exist_ok=True)

# Create a SummaryWriter instance
writer = SummaryWriter(log_dir=output_dir)

# Add the model to the SummaryWriter
# input_tensor1 = torch.randn(8, 3, 256, 256).to(DEVICE)
# input_tensor2 = torch.randn(8, 3, 256, 256).to(DEVICE)
# writer.add_graph(model, [input_tensor1, input_tensor2])

for epochs in range(MAX_EPOCH):
    torch.cuda.empty_cache()
    print('\nEpoch: {}'.format(epochs))
    model.zero_grad()
    optimizer.zero_grad()

    train_logs = train_epoch.run(train_loader)
    torch.cuda.empty_cache()
    with autocast():  # 半精度训练 torch原生支持的amp
        valid_logs = valid_epoch.run(valid_loader)

    scheduler_steplr.step()
    # torch.cuda.empty_cache()

    for key in train_metrics_history.keys():
        train_metrics_history[key].append(train_logs[key])

    for key in valid_metrics_history.keys():
        valid_metrics_history[key].append(valid_logs[key])

    for key in train_metrics_history.keys():
        fill_missing_values(train_metrics_history, key)
        fill_missing_values(valid_metrics_history, key)

    # Add the loss and accuracy to the SummaryWriter
    writer.add_scalar('train/loss', train_logs['cross_entropy_loss'], epochs)
    writer.add_scalar('train/accuracy', train_logs['accuracy'], epochs)
    writer.add_scalar('train/fscore', train_logs['fscore'], epochs)
    writer.add_scalar('train/precision', train_logs['precision'], epochs)
    writer.add_scalar('train/recall', train_logs['recall'], epochs)
    writer.add_scalar('train/iou', train_logs['iou_score'], epochs)

    writer.add_scalar('valid/loss', valid_logs['cross_entropy_loss'], epochs)
    writer.add_scalar('valid/accuracy', valid_logs['accuracy'], epochs)
    writer.add_scalar('valid/fscore', valid_logs['fscore'], epochs)
    writer.add_scalar('valid/precision', valid_logs['precision'], epochs)
    writer.add_scalar('valid/recall', valid_logs['recall'], epochs)
    writer.add_scalar('valid/iou', valid_logs['iou_score'], epochs)

    # # Add the model's weights and biases to the SummaryWriter
    # for name, param in model.named_parameters():
    #     writer.add_histogram('weights/' + name, param, epochs)
    
    # Save metrics history to txt files
    with open(os.path.join(output_dir, 'epoch_metrics_2.txt'), 'a') as epoch_file_a:
        epoch_file_a.write(f'Epoch {epochs}\n')
        epoch_file_a.write('Train Metrics: ')
        epoch_file_a.write(json.dumps(train_logs))
        epoch_file_a.write('\n')
        epoch_file_a.write('Valid Metrics: ')
        epoch_file_a.write(json.dumps(valid_logs))
        epoch_file_a.write('\n')

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        torch.save(model, os.path.join(output_dir, "e50_model_2.pth"))
        print('Model saved!')

    torch.cuda.empty_cache()

# Close the SummaryWriter
writer.close()
