# Detecting Motion to Modulate Sound live

This project was done for an art exhibition at the Hochschule Niederrhein in 
Krefeld, Germany.
It is highly specific to this exhibition, so it includes some code that might
not be very beautiful, but served the purpose.

You can view a short demonstration in [this video](https://youtu.be/IZAMerXr5zM "YouTube").
 
## What is happening:
Basically it detects the movement in front of the camera and calculates a 
'movement score'. This movement score is then used to modify various sounds
that are playing. I split up the detection in left and right so each side could 
modify a different sound. For the exhibition I used one base sound that was 
always playing, one synthesizer sound for each side and a kick that reacted to 
both sides.

My initial plan was to modulate the frequency of each sound, but at the end it 
sounded better to just modulate the amplitude of the sounds, so the sounds 
would kinda rise up with motion being detected.

The motion detection is fairly simple. A certain amount of frames is stored in
a queue. The current frame is compared to the oldest frame by calculating the 
pixel difference. This difference is processed a bit and then contours are 
identified to calculate the area of moved 'objects'. This is used as the 
movement score. The current frame gets pushed into the queue and the oldest is 
popped.

This way of storing frames results in a nice effect, since every frames gets
used twice. One time as the freshest frame and one time as the oldest. 
Therefore the detected motion kinda repeats itself one time in this short time
period.

##
For playing and modulating sound at the same time I used the 
[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/ "PyAudio Doc") 
project. The 'Callback Mode' in PyAudio enables you to specifically handle the 
upcoming audio chunks. 

The motion detection is based on [this](https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/) 
example by Adrian Rosebrock.
