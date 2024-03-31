## Harder Scence

Using the settings from last section, and training for 3000 iterations, we obtain a Mean PSNR or $18.418$ and a SSIM of $0.664$.

If we change the mean initialization, starting learning rates, incorporate an exponential learning rate scheduler, and incorporate an SSIM loss, we are
able to achieve both higher PSNR of $22.480$ and higher SSIM of $0.837$.

Specifically, based on our rough understanding of the scene, we know that that most of the objects are small in terms of height -- but they collectively
cover a large planar region. Thus, initializing the means with an anisotropic Gaussian which has more variation along this planar region, and less along
the perpendicular to this region (the vertical component of these objects), is a better initialization.

Having a high learning rate initially, particularly for parameters like the ``pre_act_scales`` 
