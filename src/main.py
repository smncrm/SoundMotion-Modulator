from imutils.video import VideoStream
import imutils
import cv2
from src import movement_service as ms
from src import sound_service as ss
from collections import deque
# ------------------------------- audio -------------------------------
import pyaudio
import numpy as np
import wave
import time

base = ss.create_sound("sounds/Motion_Base.wav")
kick = ss.create_sound("sounds/Motion_Kick.wav")
kick_add = ss.create_sound("sounds/Motion_KickAdd.wav")
add1 = ss.create_sound("sounds/Motion_Add2036.wav")
add2 = ss.create_sound("sounds/Motion_AddBeatique.wav")

sounds = {
    "base": base,
    "kick": kick,
    "kick_add": kick_add,
    "add1": add1,
    "add2": add2,
}

# instantiate PyAudio
p = pyaudio.PyAudio()

# movement values to modify the sound
movement = ms.Movement(0, 0, 0)
old_movement = ms.Movement(0, 0, 0)

# callback_count used to jump back in sample
nframes = 1024
callback_count = 0
callback_limit = base["wave"].getnframes() // nframes - 1000  # TEST THIS !!


def callback(in_data, frame_count, time_info, status):
    global callback_count
    global data
    global old_movement

    callback_count = callback_count + 1
    if callback_count == callback_limit:
        for s in sounds:
            sounds[s]["wave"].rewind()
        callback_count = 0

    for s in sounds:
        sounds[s]["data"] = ss.read_new_segment(sounds[s]["wave"], nframes)

    # # modify the sound here
    new_movement = ms.process_new_movement(movement, old_movement)

    add_data_all = ss.calculate_weighted_segment(old_movement.all,
                                                 new_movement.all, nframes,
                                                 sounds["kick"]["data"])
    add_data_left = ss.calculate_weighted_segment(old_movement.left,
                                                  new_movement.right, nframes,
                                                  sounds["add1"]["data"])
    add_data_right = ss.calculate_weighted_segment(old_movement.right,
                                                   new_movement.right, nframes,
                                                   sounds["add2"]["data"])

    data = 2 * sounds["base"]["data"]
    data = data + 3 * add_data_all
    data = data + add_data_left
    data = data + add_data_right

    if movement.all > 0.4:
        data = data + ss.calculate_weighted_segment(old_movement.all,
                                                    new_movement.all, nframes,
                                                    sounds["kick_add"]["data"])

    data_left, data_right = data[0::2], data[1::2]
    ns = np.column_stack((data_left, data_right)).ravel().astype(np.int16)

    old_movement = new_movement
    res_data = ns.tostring()

    return res_data, pyaudio.paContinue


# open stream using callback
stream = p.open(format=p.get_format_from_width(base["wave"].getsampwidth()),
                channels=base["wave"].getnchannels(),
                rate=base["wave"].getframerate(),
                output=True,
                stream_callback=callback)

# start the stream (4)
stream.start_stream()
# ----------------------------- end audio -----------------------------

# ------------------------------- init --------------------------------
min_area = 2000
scaled_width = 700

# start webcam stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

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

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=scaled_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store the frame
    frames.append(gray)

    # make sure the deque is filled first
    if count <= trackedFrames:
        frame_size = frame.shape[0] * frame.shape[1]
        count = count + 1
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(frames.popleft(), gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    frameDelta_left = frameDelta[:, 0:(scaled_width // 2)].copy()
    frameDelta_right = frameDelta[:, (scaled_width // 2):scaled_width].copy()

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh_left = thresh[:, 0:(scaled_width // 2)].copy()
    thresh_right = thresh[:, (scaled_width // 2):scaled_width].copy()

    cnts_left = cv2.findContours(thresh_left.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cnts_right = cv2.findContours(thresh_right.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    cnts_left, cnts_right = imutils.grab_contours(cnts_left), \
                            imutils.grab_contours(cnts_right)

    movements = [0, 0]
    for i in range(2):
        cnts = [cnts_left, cnts_right][i]
        area = 0
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            contour_area = cv2.contourArea(c)
            if contour_area < min_area:
                continue

            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.drawContours(frame, c, -1, (0, 255, 0), 2)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = area + contour_area

        movements[i] = round(area / frame_size, 4)

    movement = ms.Movement(sum(movements), 2 * movements[0],
                           2 * movements[1])

    # draw the text and timestamp on the frame
    cv2.putText(frame, "{}".format(movement.left), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "{}".format(movement.right), (600, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("VideoStream", frame)

    frameDelta_left = imutils.resize(frameDelta_left, width=450)
    frameDelta_right = imutils.resize(frameDelta_right, width=450)
    cv2.imshow("FrameDelta Left", frameDelta_left)
    cv2.imshow("FrameDelta right", frameDelta_right)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop()
cv2.destroyAllWindows()
# --------------------------- end main loop ---------------------------

# -------------------------- audio clean up ---------------------------
# stop stream
stream.stop_stream()
stream.close()
base.close()
kick.close()
kick_add.close()
add1.close()
add2.close()

# close PyAudio
p.terminate()
# ------------------------ end audio clean up -------------------------
