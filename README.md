# SSD_keras_restnet

I change the SSD300 base net from VGG16 to ResNet.

Base code from https://github.com/rykov8/ssd_keras.

TRAIN:
just run train_SSD.py in SSD_ResNet .

in train_SSD.py
"model = SSD_resnet.resnet_68(input_shape, num_classes=NUM_CLASSES)" is use resnet.
"model = SSD300(input_shape, num_classes=NUM_CLASSES)" is use VGG16.

TODO:
create SSD512 net and change add more layers in resnet.
