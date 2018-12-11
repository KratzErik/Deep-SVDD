# Compare reconstructions and inputs
import cv2

dataset = 'prosivic'
exp_name = 'common_arhitecture_181210'
inputs = '/home/erik/Github/Deep-SVDD/log/%s/%s/ae_inputs.png'%(dataset,exp_name)
recons = '/home/erik/Github/Deep-SVDD/log/%s/%s/ae_reconstructions.png'%(dataset,exp_name)

print(inputs)
print(recons)

x = cv2.imread(inputs)
r = cv2.imread(recons)
diff = x-r


def print_img(img):
    print("Max: %.5f\nMin: %.5f\nMean: %.5f\n"%(img.max(), img.min(),img.mean()))

print_img(x)
print_img(r)
