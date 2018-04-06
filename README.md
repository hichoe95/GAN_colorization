# GAN_colorization
pytorch GAN_colorzation

These have Trial_1 ~ Final version.

Trial_1 ~ Trial_3 has only training part, but Final has full code(it has test result)

## Training

Data : kaggle cat vs dog images(25000, 12500 cats, 12500 dogs).
training time : about 2h 30m


## Trial 1

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2012.41.04.png?raw=true" width="50%">

It has only noise.. without any shape

## Trial 2

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.37.12.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.48.png?raw=true" width="50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.41.png?raw=true" width="50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.34.png?raw=true" width="50%">

It seems a little successful. But its image is ambiguous.

## Trial 3

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.44.25.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.42.55.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.42.23.png?raw=true" width = "50%">

I just added encoded output to decoded output to remember original image's shape in Generator.

Now, we can recognize colorized cat and dog image.

If I have some more try, I will get perfect color image... 


## Final

final code conclude testing.

### Generator

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/Untitled%20Diagram.jpg?raw=true" width = "50%">


### Result

it is result which used test images (not training image)
In order,
input (black and white image),
original color image,
output (generated by Generator),

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/1.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/2.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/3.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/4.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/5.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/6.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/7.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/8.png?raw=true" width = "50%">
<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/9.png?raw=true" width = "50%">

The result is not bad but I need to study more.

## Reference

https://taeoh-kim.github.io/blog/gan을-이용한-image-to-image-translation-pix2pix-cyclegan-discogan/
https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan

## My Blog

http://hichoe95.tistory.com
