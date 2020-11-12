import time
import os 

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import transforms

class StyleTransfer:
    """
    Creates an object of StyleTransfer class.
    Use predict or predict_hr methods to make a style transfer to image from another image.
    """
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.vgg = VGG()
        load_vgg_weights(self.vgg)
        for param in self.vgg.parameters():
            param.requires_grad = False
        if self.cuda:
            self.vgg.cuda()
    
    @staticmethod
    def load_img(image):
        assert Image.isImageType(image) or isinstance(image,str) or isinstance(image,np.ndarray)
        img = image
        if isinstance(image,str):
            img = Image.open(image)
        elif isinstance(image,np.ndarray):
            img = Image.fromarray(image)
        if np.array(img).shape[2] == 4:
            img = img.convert('RGB')
        return img
    
    def img2tensor(self,image,resize):
        prep = transforms.Compose([transforms.Resize(resize),
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                   transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #imagenet mean
                                                        std=[1,1,1]),
                                   transforms.Lambda(lambda x: x.mul_(255)),
                                  ])
        return Variable(prep(image).unsqueeze(0).to(self.device))
    
    def tensor2img(self,tensor):
        postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                   transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                        std=[1,1,1]),
                                   transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                                   ])
        postpb = transforms.Compose([transforms.ToPILImage()])
        t = postpa(tensor.data[0].cpu().squeeze())
        t[t>1] = 1    
        t[t<0] = 0
        img = postpb(t)
        return img
        
    def predict_iterator(self,content,style,iters=500,transfer_color=True,scale_img=1.0,
                    print_every=0,yield_every=100,_opt_img=None):
        """
        Takes "content" and "style" images and applies style of second image to the first image.
        This method yields images every "yield_every" iterations during the process of style transfer.
        
        Yields images of PIL.Image type.

        Required parameters:
        --------------------
        content : string(path to image), PIL.Image, or numpy.ndarray
            Image to apply style
        style : string(path to image), PIL.Image, or numpy.ndarray
            Image to take style from

        Additional parameters:
        --------------------
        iters : integer
            number of iterations to produce style transfer, default 500
        transfer_color : boolean
            change content image colors to the style image colors, default True
        scale_img : float
            resize the content image size in percents, default 1.0
        print_every: integer
            print stats during style transfer, set 0 to disable printing, default 0
        yield_every: integer
            return images during style transfer, default 100
        """
        content_img = self.load_img(content)
        style_img = self.load_img(style)
        if transfer_color:
            content_arr = np.array(content_img)/255
            style_arr = np.array(style_img)/255
            content_img = Image.fromarray((match_color(content_arr,style_arr)*255).astype(np.uint8))
            
        imsize = [int(i) for i in np.array(np.array(content_img).shape[0:2])*scale_img]
        content_img = self.img2tensor(content_img,imsize)
        style_img = self.img2tensor(style_img,imsize)
        opt_img = Variable(content_img.data.clone(), requires_grad=True)
        
        style_layers = ['r11','r21','r31','r41','r51'] 
        content_layers = ['r42']
        style_weights = [1e3/(n**2) for n in [64,128,256,512,512]]
        content_weights = [1]
        
        assert (len(style_layers) == len(style_weights)) and (len(content_layers) == len(content_weights))
        
        style_targets = [gram_matrix(A).detach() for A in self.vgg(style_img, style_layers)]
        content_targets = [A.detach() for A in self.vgg(content_img, content_layers)]
        targets = style_targets + content_targets
        weights = style_weights + content_weights
        
        loss_layers = style_layers + content_layers
        loss_fns = [StyleLoss(i) for i in style_targets] + [ContentLoss(i) for i in content_targets]
        if self.cuda:
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        
        if _opt_img is not None:
            opt_img = Variable(_opt_img.unsqueeze(0).to(self.device), requires_grad=True)
        optimizer = optim.LBFGS([opt_img])
        
        iteration = 0
        while iteration <= iters:
            out = self.vgg(opt_img, loss_layers)
            layer_losses = [weights[c] * loss_fns[c](tens) for c,tens in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward(retain_graph=True)

            optimizer.step(lambda: loss)
            optimizer.zero_grad()
            
            if print_every and iteration % print_every == 0:
                print("iteration: {} loss: {}".format(iteration, loss.data))
            if iteration % yield_every == 0:
                yield self.tensor2img(opt_img.clone())
            iteration += 1
            
        out_img = self.tensor2img(opt_img)
        yield out_img
        
    def predict(self,content,style,**kwargs):
        """
        Takes "content" and "style" images and applies style of second image to the first image.
        Same as predict_iterator method, but returns only one final image of PIL.Image type.\
        """
        image = None
        for image in self.predict_iterator(content,style,**kwargs):
            pass
        return image
    
    def predict_hr(self,content,style,hr_iters=400,hr_scale=1.0,scale_img=1.0,iters=500,**kwargs):
        """
        Takes "content" and "style" images and applies style of second image to the first image.
        First it makes simple style transfer, then resizes the resulting image to "hr_scale" relative to original\
        content image scale and makes style transfer with the resulting image. That provides better image quality.\
        Parameters the same as parameters of predict_iterator method.
        
        Returns a tuple of PIL images: (img, img_hr), there img - simple style transfer result and img_hr - second style transfer result
        
        Additional parameters:
        --------------------
        hr_scale : float
            scale style transfer image secondly (relative to original content image), default 1.0
        hr_iters : integer
            num of iterations to produce second style transfer, default 400
        """
        img = self.predict(content,style,scale_img=scale_img,iters=iters,**kwargs)
        content_img = self.load_img(content)
        imsize_hr = [int(i*hr_scale) for i in np.array(content_img).shape[0:2]]
        prep_hr = transforms.Compose([transforms.Resize(imsize_hr),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
        img_hr = self.predict(content,style,scale_img=hr_scale,iters=hr_iters,
                                   _opt_img=prep_hr(img),**kwargs)
        return (img,img_hr)
        