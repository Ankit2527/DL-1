import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os 
%matplotlib inline
from matplotlib.pyplot import figure
import itertools

#Path for the json files for models with augmentation
path = './CNN/jsons'

#Open json files for models with augmentation
dict_1 = {}
models = []
for file in os.listdir(path):
    model = file.split('.json')[0]
    file = os.path.join(path, file)
    file_new = open(file)
    data = json.load(file_new)
    dict_1[str(model)] = data
    models.append(model)
dict_2 = dict_1

gaussian_noise = [dict_2['resnet18']['gaussian_noise_transform_1'], dict_2['resnet18']['gaussian_noise_transform_2'], dict_2['resnet18']['gaussian_noise_transform_3'], dict_2['resnet18']['gaussian_noise_transform_4'], dict_2['resnet18']['gaussian_noise_transform_5']],
gaussian_blur = [dict_2['resnet18']['gaussian_blur_transform_1'], dict_2['resnet18']['gaussian_blur_transform_2'], dict_2['resnet18']['gaussian_blur_transform_3'], dict_2['resnet18']['gaussian_blur_transform_4'], dict_2['resnet18']['gaussian_blur_transform_5']],
contrast_reduction = [dict_2['resnet18']['contrast_transform_1'], dict_2['resnet18']['contrast_transform_2'], dict_2['resnet18']['contrast_transform_3'], dict_2['resnet18']['contrast_transform_4'], dict_2['resnet18']['contrast_transform_5']],
jpeg_compression = [dict_2['resnet18']['jpeg_transform_1'], dict_2['resnet18']['jpeg_transform_2'], dict_2['resnet18']['jpeg_transform_3'], dict_2['resnet18']['jpeg_transform_4'], dict_2['resnet18']['jpeg_transform_5']],

gaussian_noise = list(itertools.chain.from_iterable(gaussian_noise))
gaussian_blur = list(itertools.chain.from_iterable(gaussian_blur))
contrast_reduction = list(itertools.chain.from_iterable(contrast_reduction))
jpeg_compression = list(itertools.chain.from_iterable(jpeg_compression))

#Load jsons files for models with no augmentation
with open('./CNN/resnet18_no_augmentation.json') as f:
    resnet18_no_augmentation = json.load(f)
with open('./CNN/resnet18_no_augmentation.json') as f:
    resnet34_no_augmentation = json.load(f)
with open('./CNN/resnet18_no_augmentation.json') as f:
    vgg11_no_augmentation = json.load(f)
with open('./CNN/resnet18_no_augmentation.json') as f:
    vgg11_bn_no_augmentation = json.load(f)
with open('./CNN/resnet18_no_augmentation.json') as f:
    densenet121_no_augmentation = json.load(f)
    
resnet18_no_augmentation = resnet18_no_augmentation['gaussian_noise_transform_0']
resnet34_no_augmentation = resnet34_no_augmentation['gaussian_noise_transform_0']
vgg11_no_augmentation = vgg11_no_augmentation['gaussian_noise_transform_0']
vgg11_bn_no_augmentation = vgg11_bn_no_augmentation['gaussian_noise_transform_0']
densenet121_no_augmentation = densenet121_no_augmentation['gaussian_noise_transform_0']
    
#Plotting for different corruption functions on ResNet-18
plt.figure(figsize=(15, 7), dpi = 400) 
x = [1, 2, 3, 4, 5]
resnet_18_no_augmentation = [resnet18_no_augmentation]*5
resnet_18_no_augmentation_new = [100 - x for x in resnet_18_no_augmentation]
gaussian_noise_new = [100 - x for x in gaussian_noise]
gaussian_blur_new = [100 - y for y in gaussian_blur]
contrast_new = [100 - z for z in contrast_reduction]
jpeg_transform_new = [100 - t for t in jpeg_compression]
plt.plot(x, resnet_18_no_augmentation_new, label='Without Corruption')
plt.plot(x, gaussian_noise_new, label='Gaussian Noise')
plt.plot(x, gaussian_blur_new, label='Gaussian Blur')
plt.plot(x, contrast_new, label='Contrast Reduction')
plt.plot(x, jpeg_transform_new, label='JPEG Compression')
plt.xlabel('Severity Values ranging from 1-5')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot for different corruption functions on ResNet-18')
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
path = './' 
fname = 'Accuracy_plot_for_different_corruption_functions_on_ResNet-18' + '.png'
fname = os.path.join(path, fname)
plt.savefig(fname,bbox_inches='tight')
plt.show() 

dummy = {}
dummy_2 = {}
num = 0
m = 0
for keys_1, values_1  in dict_1.items():
    if 'resnet18' in keys_1:
        d4 = {}
        temp_1 = dict_1[keys_1]
        for keys_2, values_2 in temp_1.items():
            values_2 = values_2
            num += values_2
            m = m + 1
            if (m % 5 == 0):
                temp = num
                dummy_2[keys_2] = temp
                num= 0
            dummy[keys_1] = dummy_2
    dummy_2 = {}
    
dummy_2 = {}
dict_3 = {}
num = 0
m = 0
for keys_1, values_1  in dict_1.items():
    d4 = {}
    temp_1 = dict_1[keys_1]
    for keys_2, values_2 in temp_1.items():
        values_2 = values_2
        num+=  values_2
        m = m + 1
        if (m % 5 == 0 ):
            temp = num / dummy ['resnet18'][str(keys_2)]
            dummy_2[keys_2] = temp
            num= 0
    dict_3[keys_1] = dummy_2
    dummy_2 = {}

models.remove('resnet18')

plot_data_ce = pd.DataFrame({
    "gaussian_noise_transform": [dict_3[models[0]]['gaussian_noise_transform_5'] , dict_3[models[1]]['gaussian_noise_transform_5'], dict_3[models[2]]['gaussian_noise_transform_5'], dict_3[models[3]]['gaussian_noise_transform_5']],
    "gaussian_blur_transform": [dict_3[models[0]]['gaussian_blur_transform_5'] , dict_3[models[1]]['gaussian_blur_transform_5'], dict_3[models[2]]['gaussian_blur_transform_5'], dict_3[models[3]]['gaussian_blur_transform_5']],
    "contrast_reduction": [dict_3[models[0]]['contrast_transform_5'] , dict_3[models[1]]['contrast_transform_5'], dict_3[models[2]]['contrast_transform_5'], dict_3[models[3]]['contrast_transform_5']],
    "jpeg_compression" : [dict_3[models[0]]['jpeg_transform_5'] , dict_3[models[1]]['jpeg_transform_5'], dict_3[models[2]]['jpeg_transform_5'], dict_3[models[3]]['jpeg_transform_5']],
    }, index=models)
plt.figure(figsize=(15, 7), dpi = 400) 
plot_data_ce.plot(kind = "bar")
plt.title("CE Plot")
plt.ylabel("CE Value")
plt.xlabel('Models')
plt.legend(loc = 'best', bbox_to_anchor = (1, 0.5))
plt.savefig("CE_plot.png", bbox_inches = 'tight')
plt.margins(50, 0.1)

no_augmentation = {}
no_augmentation = {'resnet18': [resnet18_no_augmentation]*5,
 'resnet34': [resnet34_no_augmentation]*5,
 'vgg11': [vgg11_no_augmentation]*5,
 'vgg11_bn': [vgg11_bn_no_augmentation]*5,
 'densenet121': [densenet121_no_augmentation]*5}

dict_3 = {}
dummy_2 = {}
num = 0
m = 0
for keys_1, values_1  in dict_1.items():
    d4 = {}
    temp_1 = dict_1[keys_1]
    for keys_2, values_2 in temp_1.items():
        values_2 = values_2
        num+=  values_2
        m = m + 1
        if (m % 5 == 0 ):
            num= num- np.sum(no_augmentation['resnet18'])
            denum = (dummy['resnet18'][str(keys_2)]) - np.sum(no_augmentation['resnet18'])
            temp = num / denum
            dummy_2[keys_2] = temp
            num= 0
    dict_3[keys_1] = dummy_2
    dummy_2 = {}

plot_data_rce = pd.DataFrame({
    "gaussian_noise_transform": [dict_3[models[0]]['gaussian_noise_transform_5'] , dict_3[models[1]]['gaussian_noise_transform_5'], dict_3[models[2]]['gaussian_noise_transform_5'], dict_3[models[3]]['gaussian_noise_transform_5']],
    "gaussian_blur_transform": [dict_3[models[0]]['gaussian_blur_transform_5'] , dict_3[models[1]]['gaussian_blur_transform_5'], dict_3[models[2]]['gaussian_blur_transform_5'], dict_3[models[3]]['gaussian_blur_transform_5']],
    "contrast_reduction": [dict_3[models[0]]['contrast_transform_5'] , dict_3[models[1]]['contrast_transform_5'], dict_3[models[2]]['contrast_transform_5'], dict_3[models[3]]['contrast_transform_5']],
    "jpeg_compression" : [dict_3[models[0]]['jpeg_transform_5'] , dict_3[models[1]]['jpeg_transform_5'], dict_3[models[2]]['jpeg_transform_5'], dict_3[models[3]]['jpeg_transform_5']],
    }, index=models)
plt.figure(figsize=(15, 7), dpi = 400) 
plot_data_rce.plot(kind = "bar")
plt.title("RCE")
plt.ylabel("RCE Value")
plt.xlabel('Models')
plt.legend(loc = 'best', bbox_to_anchor = (1, 0.5))
plt.savefig("RCE_plot.png", bbox_inches = 'tight')
plt.margins(50, 0.1)
