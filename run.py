import argparse
from PIL import Image
from style_transfer import StyleTransfer

desc = "PyTorch implementation of Neural Style Transfer with color control."
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--content', type=str, 
                    help='Path to content image', required = True)
parser.add_argument('--style', type=str, 
                    help='Path to style Image', required = True)
parser.add_argument('--output', type=str, default='result.jpg', 
                    help='File path of output image')
parser.add_argument('--iters', type=int, default=500, 
                    help='The number of iterations to run')
parser.add_argument('--transfer_color', type=bool, default=True, 
                    help='Transfer colors of the style image')
parser.add_argument('--scale_img', type=float, default=1.0, 
                    help='Scaling factor of the image size')
parser.add_argument('--print_every', type=int, default=0, 
                    help='The number of iterations to show stats (set 0 to not show stats)')

parser.add_argument('--content_layers', nargs='+', type=str, default=['r42'], 
                    help='VGG19 layers used for content loss')
parser.add_argument('--style_layers', nargs='+', type=str, default=['r11','r21','r31','r41','r51'],
                    help='VGG19 layers used for style loss')

parser.add_argument('--content_weights', nargs='+', type=float, default=[1.0], 
                    help='Content loss weights for each content layer')
parser.add_argument('--style_weights', nargs='+', type=float, default=[0.2,0.2,0.2,0.2,0.2],
                    help='Style loss weights for each style layer')

args = parser.parse_args()

if __name__=='__main__':
    st = StyleTransfer()
    img = st.predict(args.content,args.style,iters=args.iters,transfer_color=args.transfer_color,
                     scale_img=args.scale_img,print_every=args.print_every,content_layers=args.content_layers,
                     style_layers=args.style_layers,content_weights=args.content_weights,
                     style_weights=args.style_weights)
    img.save(args.output)
    print('Saved to {}'.format(args.output))