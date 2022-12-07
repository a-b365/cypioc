How It Works?

Our model follows the Plain CNN architecture. The network is made of stacked convolutional layers. Each block seen in the figure is composed of two to three convolutional layers followed by a rectified-linear activation unit. Between blocks, a batch normalization layer is inserted to help prevent exploding or vanishing gradient problems and speed up convergence.

The figure gives the general overview of our network.

![architecture](https://user-images.githubusercontent.com/63753115/206283214-4a3ce889-eb56-4b90-a858-ba05e9f474f8.png)

CIE lab color space

Our model uses CIE Lab color to separate color information to a separate channel which can be added to the grayscale image when training with the model. The CIE in CIELAB is the abbreviation for the International Commission on Illumination’s French name. The letters L*, a* and b* represent each of the three values the CIELAB color space uses to measure objective color and calculate color differences. An image in the Lab color space consists of one channel for achromatic luminance (L) and two-color channels (ab). The ‘a’ channel controls hues between green and red, while the ‘b’ channel has control over hues between blue and yellow.

Screenshots of system
![Screenshot (10)](https://user-images.githubusercontent.com/63753115/206284943-2ea5fa6d-2438-40a5-bed3-030969ba1435.png)
![Screenshot (11)](https://user-images.githubusercontent.com/63753115/206284968-13bcf445-05cd-48c4-9e54-59b0b7216d0b.png)
