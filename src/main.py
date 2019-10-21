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

    data = sounds["base"]["data"]
    data = data + 2 * add_data_all
    data = data + add_data_left
    data = data + add_data_right

    if movement.all > 0.4:
        data = data + ss.calculate_weighted_segment(old_movement.all,
                                                    new_movement.all,
                                                    nframes,
                                                    2 * sounds["kick_add"][
                                                        "data"])
    data = 4 * data
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

    frame = ms.process_frame(vs.read(), config_video)
    frames.append(frame)

    count = count + 1

config_video["frame area"] = frame.shape[0] * frame.shape[1]

# ----------------------------- main loop -----------------------------
while True:
    frame = ms.process_frame(vs.read(), config_video)
    frames.append(frame)

    movement, frame_left, frame_right = ms.calculate_movement(frames,
                                                              config_video)

    frame_left = imutils.resize(frame_left, width=470)
    frame_right = imutils.resize(frame_right, width=470)
    cv2.imshow("FrameDelta Left", frame_left)
    cv2.imshow("FrameDelta right", frame_right)

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
