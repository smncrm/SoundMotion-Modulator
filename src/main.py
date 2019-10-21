import imutils
import cv2
import pyaudio
import time
import numpy as np
from imutils.video import VideoStream
from src import movement_service as ms
from src import sound_service as ss
from collections import deque

# ------------------------------- audio -------------------------------
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
    data = data + 0.7 * add_data_right

    if movement.all > 0.4:
        data = data + ss.calculate_weighted_segment(old_movement.all,
                                                    new_movement.all, nframes,
                                                    sounds["kick_add"]["data"])

    data_left, data_right = data[0::2], data[1::2]
    ns = np.column_stack((data_left, data_right)).ravel().astype(np.int16)

    old_movement = new_movement
    res_data = ns.tostring()

    return res_data, pyaudio.paContinue


stream = p.open(format=p.get_format_from_width(base["wave"].getsampwidth()),
                channels=base["wave"].getnchannels(),
                rate=base["wave"].getframerate(),
                output=True,
                stream_callback=callback)
stream.start_stream()

# ----------------------------- init video ----------------------------
config_video = {
    "frame width": 700,
    "min area": 2000,
    "frame area": 0,
    "tracked frames": 50
}

vs = VideoStream(src=0).start()
time.sleep(2.0)

frame = None
frames = deque()

# count the frames that have been processed since the firstFrame was updated
count = 0
# the movement value for every frame is stored in this list
areas = deque()

# make sure the frames deque is full before motion detection
while count <= config_video["tracked frames"]:
    frame = vs.read()

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=config_video["frame width"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # store the frame
    frames.append(gray)

    count = count + 1

config_video["frame area"] = frame.shape[0] * frame.shape[1]

# ----------------------------- main loop -----------------------------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=config_video["frame width"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    frames.append(gray)

    movement, frameDelta_left, frameDelta_right = ms.calculate_movement(frames,
                                                                        config_video)

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

    if key == ord("q"):
        break

# -------------------------- clean up ---------------------------
vs.stop()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
for s in sounds:
    sounds[s]["wave"].close()
