import click
import torch
from torchvision import models as models
import albumentations
import os
from PIL import Image
import numpy as np
import json


@click.command()
@click.argument('path')
def process(path):
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')  # включаем поддержку GPU
    if device == 'cuda:0':
        click.echo('GPU on...')
    else:
        click.echo('GPU off...')

    net = models.resnet18(pretrained=False).to(device)
    net.fc = torch.nn.Linear(512, 2).to(device)  # ставим выходной слой на 2 класса
    net.load_state_dict(torch.load('epoch_3.pth'))  # загружаю веса самой удачной эпохи 
    net.eval()
    aug = albumentations.Compose([
        albumentations.Resize(224, 224, always_apply=True),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], always_apply=True)
        ])
    click.echo('Processing...')
    db = {}
    for im_name in os.listdir(path):
        if im_name[-3:] == 'jpg':
            image = Image.open(os.path.join(path, im_name))  # загружаем фото в RAM
            image = aug(image=np.array(image))['image']  # редактируем
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # меняем каналы под формат pytorch (channels, H, W)
            image = image[np.newaxis, :, :, :]
            pred = net(torch.tensor(image, dtype=torch.float).to(device))
            pred = pred.cpu().detach()
            db[im_name] = 'male' if pred.argmax() == 0 else 'female'

    with open(os.path.join(path, 'process_results.json'), 'w') as fp:
        json.dump(db, fp)

    click.echo('Well done!')
    click.echo(f'Saved as process_results.json in {path}')


if __name__ == '__main__':
    process()