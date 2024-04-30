#cypioc(color your pictures in on click)
How It Works?

Our model follows the Plain CNN architecture. The network is made of stacked convolutional layers. Each block seen in the figure is composed of two to three convolutional layers followed by a rectified-linear activation unit. Between blocks, a batch normalization layer is inserted to help prevent exploding or vanishing gradient problems and speed up convergence.

The figure gives the general overview of our network.

![architecture](https://user-images.githubusercontent.com/63753115/206283214-4a3ce889-eb56-4b90-a858-ba05e9f474f8.png)

CIE lab color space

Our model uses CIE Lab color to separate color information to a separate channel which can be added to the grayscale image when training with the model. The CIE in CIELAB is the abbreviation for the International Commission on Illumination’s French name. The letters L*, a* and b* represent each of the three values the CIELAB color space uses to measure objective color and calculate color differences. An image in the Lab color space consists of one channel for achromatic luminance (L) and two-color channels (ab). The ‘a’ channel controls hues between green and red, while the ‘b’ channel has control over hues between blue and yellow.

Screenshots of system
![Screenshot (12)](https://user-images.githubusercontent.com/63753115/206371874-36215f13-d561-4b40-a6aa-d817c07c908a.png)
![Screenshot (13)](https://user-images.githubusercontent.com/63753115/206371883-50425056-eaa7-4892-9c35-9fd547d55e3f.png)
