# Detecting Motion to Modulate Sound live

This project was done for an art exhibition at the Hochschule Niederrhein in 
Krefeld, Germany.
It is highly specific to this exhibition, so it includes some code that might
not be very beautiful, but served the purpose.

##
For playing and modulating sound at the same time I used the 
[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/ "PyAudio Doc") 
project. The 'Callback Mode' in PyAudio enables you to specifically handle the 
upcoming audio chunks. 

The motion detection is based on [this](https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/) 
example by Adrian Rosebrock.
 
