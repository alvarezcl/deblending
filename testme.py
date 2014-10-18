import galsim
import numpy as np
import deblend

bd=galsim.BaseDeviate(1)

def create_blend(peak1, peak2):
    gal1 = galsim.Gaussian(fwhm=1.2, flux=2000.0).shift(peak1).shear(e1=0.1, e2=0.3)
    gal2 = galsim.Gaussian(fwhm=1.8, flux=2500.0).shift(peak2).shear(e1=-0.1, e2=-0.4)
    proto_image = galsim.ImageD(49, 49, scale=0.2)
    image1 = gal1.drawImage(image=proto_image, method='phot', rng=bd)
    #image1 = gal1.drawImage(image=proto_image, method='fft')
    image1.array[np.where(image1.array < 0)] = 0.
    proto_image = galsim.ImageD(49, 49, scale=0.2)
    image2 = gal2.drawImage(image=proto_image, method='phot', rng=bd)
    #image2 = gal2.drawImage(image=proto_image, method='fft')
    image2.array[np.where(image2.array < 0)] = 0.
    return image1+image2, [image1, image2]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    peak1 = (0.8, 0.2)
    peak2 = (-0.3, 1.26)

    peaks_pix = [[p1/0.2 for p1 in peak1],
                 [p2/0.2 for p2 in peak2]]

    blend, unblends = create_blend(peak1, peak2)

    templates, template_fractions, children = deblend.deblend(blend.array, peaks_pix)

    sum_children = children[0] + children[1]
    
    residual = blend.array - sum_children

    print blend.array.sum()
    print children[0].sum() + children[1].sum()

    fig = plt.figure(figsize=(20,11))
    ax1 = fig.add_subplot(4,3,1)
    plt.title('Original Image')
    a = ax1.imshow(blend.array, vmin=0, vmax=50)
    plt.colorbar(a)
    ax2 = fig.add_subplot(4,3,2)
    plt.title('Object 1')
    ax3 = fig.add_subplot(4,3,3)
    plt.title('Object 2')  
    ax2.imshow(unblends[0].array, vmin=0, vmax=50)
    ax3.imshow(unblends[1].array, vmin=0, vmax=50)

    ax5 = fig.add_subplot(4,3,5)
    plt.title('Symmetric Template about Peak 1')
    ax6 = fig.add_subplot(4,3,6)
    plt.title('Symmetric Template about Peak 2')
    ax5.imshow(templates[0], vmin=0, vmax=50)
    ax6.imshow(templates[1], vmin=0, vmax=50)

    ax8 = fig.add_subplot(4,3,8)
    plt.title('Template Fraction of Object 1')
    ax9 = fig.add_subplot(4,3,9)
    plt.title('Template Fraction of Object 2')    
    ax8.imshow(template_fractions[0], vmin=0, vmax=1)
    ax9.imshow(template_fractions[1], vmin=0, vmax=1)
    

    ax11 = fig.add_subplot(4,3,11)
    plt.title('Deblended Object 1')
    ax12 = fig.add_subplot(4,3,12)
    plt.title('Deblended Object 2')
    ax11.imshow(children[0], vmin=0, vmax=50)
    ax12.imshow(children[1], vmin=0, vmax=50)
    
    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,1,1)
    a = ax1.imshow(residual,interpolation='none')
    plt.colorbar(a)

    plt.show()
