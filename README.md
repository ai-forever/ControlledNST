# Pytorch-Neural-Style-Transfer
A PyTorch implementation of neural style transfer with color control described in the papers:
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)
* [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/pdf/1611.07865.pdf)
## Examples
### The Neckarfront in Tübingen, Germany
The results were obtained with the default settings except `scale_img=0.7`

<p align="center">
<img src="images/tubingen.jpg" height="192px">
<img src="images/tubingen-starry-night.jpg" height="192px">
<img src="images/tubingen-shipwreck.jpg" height="192px">

<img src="images/tubingen-kandinsky.jpg" height="192px">
<img src="images/tubingen-graffiti.jpg" height="192px">
<img src="images/tubingen-mosaic.jpg" height="192px">
</p>

When you reduce the image size, the style becames coarser. Images from left to right: `original image`, `scale_img=1.0`, `scale_img=0.4`

<p align="center">
<img src="images/tubingen.jpg" height="192px">
<img src="images/tubingen-starry-night-scale10.jpg" height="192px">
<img src="images/tubingen-starry-night-scale04.jpg" height="192px">
</p>
