import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic
from skimage.color import gray2rgb,rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import os
import csv
import sys
from torchvision import transforms, utils
from model.test import Model

def load_data(doc):
    x = []
    y = []
    #x_train = np.zeros([18,1])
    with open(doc, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            y.append(row[0])
            x.append(row[1].split())
    x.pop(0)
    y.pop(0)

    x_choose = []
    y_choose = []
    k = 0
    for i in range(len(y)):
        if k == 7:
            break
        if int(y[i]) == k:
            x_choose.append(x[i])
            y_choose.append(y[i])
            k += 1

    #normalization to 0~1
    x = np.array(x_choose,dtype=float) / 255.0
    y = np.array(y_choose,dtype=int)
    x = np.reshape(x,(x.shape[0],1,48,48))
    
    '''
    for i in range(2):
        x = np.concatenate((x,x))
        y = np.concatenate((y,y))
    '''
    #x = torch.tensor(x)
    #y = torch.tensor(y)
    
    return x,y



def predict(input):
    #print(input.shape)
    test = []
    for i in range(input.shape[0]):
        img = rgb2gray(input[i])
        img.reshape((48,48))
        test.append(img)
    test = np.array(test)
    test = test.reshape(test.shape[0],1,48,48)
    #print(test.shape)
    #exit(0)
    test = torch.tensor(test)
    #print(img.shape)
    '''test = []
    for i in range(input.shape[0]):
        test.append([0.2,5])'''

    device = torch.device('cuda')
    model = Model()
    model.eval()
    model.load_state_dict(torch.load('best.pkl'))
    model = model.to(device)
    
    pred = model(test.float().to(device))
    result = pred.detach().cpu().numpy()
    #max_index = np.argmax(result)
    #print(result[max_index],max_index)
    
    return result

def segmentation(input):
    return slic(input)

def main():
    print("Run Hw4_3 ...")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    doc = sys.argv[1]
    x_train,y_train = load_data(doc)
    '''x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    model = Model()
    
    model.eval()
    model.load_state_dict(torch.load('../test/best.pkl'))
    model = model.to(device)
    model.fc.register_forward_hook(get_activation('fc'))
    model(x_train[0].view(1,1,48,48).float().to(device))
    print(x_train_rgb[0].shape)'''

    #y_pred = model(x_train[0].view(1,1,48,48).float().to(device)
    
    explainer = lime_image.LimeImageExplainer(random_state=87)

    # Get the explaination of an image


    for i in range(x_train.shape[0]):
        explaination = explainer.explain_instance(
                                    image=x_train[i][0], 
                                    classifier_fn=predict,
                                    segmentation_fn=segmentation,
                                    batch_size=128,
                                    num_samples=1000,
                                    top_labels=7,
                                    random_seed=87
                                )

        # Get processed image
        image, mask = explaination.get_image_and_mask(
                                        label=y_train[i],
                                        positive_only=False,
                                        hide_rest=False,
                                        num_features=5,
                                        min_weight=0.0
                                    )

        # save the image
        plt.imsave('{}fig3_{}.jpg'.format(sys.argv[2],i), image)
        print('fig3_{}.jpg is completed!'.format(i))


if __name__ == "__main__": main()