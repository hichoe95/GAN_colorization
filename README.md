# GAN_colorization
pytorch GAN_colorzation

I'll update continuously...

Until now , these are incomplete result.

These only have training code. When I complete this model, I'll upload full code and test result.


### Trial 1

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2012.41.04.png?raw=true" width="50%">

It has only noise.. without any shape

### Trial 2

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.37.12.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.48.png?raw=true" width="50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.41.png?raw=true" width="50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%2010.36.34.png?raw=true" width="50%">

It seems a little successful. But its image is ambiguous.

### Trial 3

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.44.25.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.42.55.png?raw=true" width = "50%">

<img src="https://github.com/hichoe95/GAN_colorization/blob/master/result_IMG_in_training/스크린샷%202018-04-05%20오후%205.42.23.png?raw=true" width = "50%">

I just added encoded output to decoded output to remember original image's shape in Generator.

Now, we can recognize colorized cat and dog image.

If I have some more try, I will get perfect color image... 
