# self-supervised-segmentation
self-supervised segmentation of single species shapes via cycleGAN or differentiable rendering.


## What I did so far:
I cloned https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and made the network residual via changing the `forward` method of the `UnetGenerator` from
```python
   def forward(self, input):
        """Standard forward"""
        return self.model(input)
```
to
```python
    def forward(self, input):
        """Standard forward"""
        return input + self.model(input)
```
I created a new disk dataset with the provided script and trained a cycleGAN network on this dataset.

This seemed to work:
![alt text](figures/disk_cycleGAN_resunet.png "Residual Unet works")

I checked if the spatial correspondence comes from the residual structure. So I reversed the above change and repeated training. This worked, too:
![alt text](figures/disk_cycleGAN_unet.png "Just Unet works, too")

The spatial correspondence is also working for affinity maps instead of sketches:
![alt text](figures/disk-affinity_cycleGAN_unet.png "Just Unet with affinity maps works, too")

I created a separate toy set with voronoi regions that were opened morphologically and deformed elastically. The network did work good on them, too:
![alt text](figures/voronoi_cycleGAN_unet.png "Voronoi")

Since affinity maps have two channels and input and output image do need the same number of channels in the cycleGAN model, there is room for improvement. Input and output channel numbers need to be the same for cycleGAN due to the identity loss in this model. I *conjecture* that the identity loss is responsible for the spatial correspondence that now can be seen, and that I couldn´t see last time with my own implementation of cycleGAN without identity loss. I did not use the identity loss, because I was using different input and output channels.

Next, I must show, that the spatial correspondence is gone, when abolishing the identity loss.

After that, I need to introduce the spatial correspondence via a new method. One possibility would be to append an identity channel to the generator networks. This would solve the above problem, but this does not give us an interpretable insight. Having an additional loss depending on Euclidean transformation would give a new insight. This would harden the assumption, that the network needs some kind of notion of spatial correspondence. This assumption is implicitly backed into the identity plot.

Another use case would be artifacts on the noisy images, like dirt on the coverslip or in the beam path. Therefore they could be treated as different objects in separate channels.

## Further Questions:
Can we use a VAE cycleGAN and constrain some of the latent vectors to represent radius, intensity or other measurable values?
