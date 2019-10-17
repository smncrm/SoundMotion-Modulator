
from imutils.video import VideoStream
import argparse
import imutils
import cv2
from collections import deque
# ------------------------------- audio -------------------------------
import pyaudio
import numpy as np
import wave
import time

base = wave.open("Motion_Base.wav", "rb")
kick = wave.open("Motion_Kick.wav", "rb")
kick_add = wave.open("Motion_KickAdd.wav", "rb")
add1 = wave.open("Motion_Add2036.wav", "rb")
add2 = wave.open("Motion_AddBeatique.wav", "rb")

# instantiate PyAudio
p = pyaudio.PyAudio()

# movement values to modify the sound
movement = 0
old_movement = 0

# callback_count used to jump back in sample
nframes = 1024
callback_count = 0
callback_limit = base.getnframes() // nframes


def callback(in_data, frame_count, time_info, status):
    global callback_count
    global data
    global old_movement

    callback_count = callback_count + 1
    base_wave = base.readframes(nframes)
    kick_wave = kick.readframes(nframes)
    kick_add_wave = kick_add.readframes(nframes)
    add1_wave = add1.readframes(nframes)
    add2_wave = add2.readframes(nframes)

    if callback_count == callback_limit:
        base.setpos(0)
        kick.setpos(0)
        kick_add.setpos(0)
        add1.setpos(0)
        add2.setpos(0)
        callback_count = 0

    # # modify the sound here
    if movement < old_movement * 0.9:
        new_movement = old_movement * 0.9
    else:
        new_movement = movement

    base_data = np.fromstring(base_wave, dtype=np.int16)
    kick_data = np.fromstring(kick_wave, dtype=np.int16)
    kick_add_data = np.fromstring(kick_add_wave, dtype=np.int16)
    add1_data = np.fromstring(add1_wave, dtype=np.int16)
    add2_data = np.fromstring(add2_wave, dtype=np.int16)

    add_data = 3*kick_data + add1_data + add2_data
    weight = np.linspace(old_movement, new_movement, nframes * 2)
    data = 2*(base_data + 0.01*add1_data
              + np.multiply(weight, add_data))

    if movement > 0.4:
        data = data + np.multiply(weight, 3*kick_add_data)

    old_movement = new_movement

    data_left, data_right = data[0::2], data[1::2]
    #data_lf, data_rf = np.fft.rfft(data_left), np.fft.rfft(data_right)
    #nl, nr = np.fft.irfft(data_lf), np.fft.irfft(data_rf)
    ns = np.column_stack((data_left, data_right)).ravel().astype(np.int16)

    res_data = ns.tostring()

    return res_data, pyaudio.paContinue


# open stream using callback
stream = p.open(format=p.get_format_from_width(base.getsampwidth()),
                channels=base.getnchannels(),
                rate=base.getframerate(),
                output=True,
                stream_callback=callback)

# start the stream (4)
stream.start_stream()
# ----------------------------- end audio -----------------------------

# ------------------------------- init --------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=2000,
                help="minimum area size")
args = vars(ap.parse_args())

# start webcam stream
vs = VideoStream(src=1).start()
time.sleep(2.0)

# initialize the first frame in the video stream
firstFrame = None
trackedFrames = 50
frames = deque()

# count the frames that have been processed since the firstFrame was updated
count = 0
# the movement value for every frame is stored in this list
areas = deque()
# ----------------------------- main loop -----------------------------
while True:
    # grab the current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store the frame
    frames.append(gray)

    if count <= trackedFrames:
        frame_size = frame.shape[0] * frame.shape[1]
        count = count + 1
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(frames.popleft(), gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    area = 0
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        contour_area = cv2.contourArea(c)
        if contour_area < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        #(x, y, w, h) = cv2.boundingRect(c)
        cv2.drawContours(frame, c, -1, (0, 255, 0), 2)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = area + contour_area

    movement = round(area / frame_size, 4)

    # draw the text and timestamp on the frame
    cv2.putText(frame, "{}".format(movement), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break


# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
# --------------------------- end main loop ---------------------------

# -------------------------- audio clean up ---------------------------
# stop stream
stream.stop_stream()
stream.close()
base.close()
kick.close()

# close PyAudio
p.terminate()
# ------------------------ end audio clean up -------------------------