# Gaussian Splatting

## Rendering

![Views of the provided scene represented by pre-trained 3D Gaussians.](./Q1/output/q1_render_sled.gif)

## Training 3D Gaussian Representations

For this question, only L1 loss is used. The following learning rates are used:

```python
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.01, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.0025, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.00016, "name": "means"},
    ]
```

The number of iterations used for training was ~400. The final PSNR was $27.546$, and the Mean SSIM was $0.931$.

Below are the resultant GIFs:

![Turntable view of cow.](./Q1/output/cow_turntable.gif)

![Training progress for cow.](./Q1/output/q1_training_progress_cow.gif){width=500px}

## Harder Scene

Using the settings from last section, and training for 3000 iterations, we obtain a Mean PSNR or $18.418$ and a SSIM of $0.664$.

If we change the mean initialization, starting learning rates, incorporate an exponential learning rate scheduler, and incorporate an SSIM loss, we are
able to achieve both higher PSNR of $22.480$ and higher SSIM of $0.837$.

Specifically, based on our rough understanding of the scene, we know that that most of the objects are small in terms of height -- but they collectively
cover a large planar region. Thus, initializing the means with an anisotropic Gaussian which has more variation along this planar region, and less along
the perpendicular to this region (the vertical component of these objects), is a better initialization.

```python
        k = torch.Tensor([0.6,0.6,0.2]).unsqueeze(0)
        data["means"] = torch.randn((num_points, 3)).to(torch.float32) * k
```

Having a high learning rate initially, particularly for parameters like the ``pre_act_scales``, helps in optimizing
the Gaussians to fit to the scene -- many of the objects can be approximated with one big Gaussian, as they are roughly
spherical.

```python
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 1e-2, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 1e-1, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 1e-1, "name": "colours"},
        {'params': [gaussians.means], 'lr': 1e-2, "name": "means"},
    ]

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: 0.1**(i/1000))
```

Above are the learning rates and scheduler used. The default Adam optimizer was used.

![Using the learning configuration for the easy scene.](./Q1/output/q1_harder_training_final_renders.gif)

![Using the custom learning configuration.](./Q1/output/q1_harder_training_final_renders_good.gif)

![Training progress with default learning configuration.](./Q1/output/q1_harder_training_progress.gif){width=500px}

![Training progress with the new learning configuration.](./Q1/output/q1_harder_training_progress_good.gif){width=500px}

# Diffusion-guided Optimization

## SDS Loss + Image Optimization

The optimized image output for several different prompts, with guidance:

![Output with the prompt "a hamburger".](./Q2/output/image/a_hamburger_good/output.png)

![Output with the prompt "a standing corgi dog".](./Q2/output/image/a_corgi_dog/output.png)

![Output with the prompt "a cat".](./Q2/output/image/a_cat/output.png)

![Output with the prompt "a car".](./Q2/output/image/a_car/output.png)
