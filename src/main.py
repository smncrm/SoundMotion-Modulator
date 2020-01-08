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
base = ss.create_sound("sounds/Motion_Base_loud.wav")
kick = ss.create_sound("sounds/Motion_Kick_loud.wav")
kick_add = ss.create_sound("sounds/Motion_KickAdd_loud.wav")
add_left = ss.create_sound("sounds/Motion_AddPoisonPluto_loud.wav")
add_right = ss.create_sound("sounds/Motion_AddBeatique_loud.wav")

sounds = {
    "base": base,
    "kick": kick,
    "kick_add": kick_add,
    "add_left": add_left,
    "add_right": add_right,
}

p = pyaudio.PyAudio()

movement = ms.Movement(0, 0, 0)
old_movement = ms.Movement(0, 0, 0)

nframes = 1024
callback_count = 0
callback_limit = base["wave"].getnframes() // nframes - 1000


def callback(in_data, frame_count, time_info, status):
    global callback_count
    global old_movement

    callback_count = callback_count + 1
    if callback_count == callback_limit:
        for s in sounds:
            sounds[s]["wave"].rewind()
        callback_count = 0

    for s in sounds:
        sounds[s]["data"] = ss.read_new_segment(sounds[s]["wave"], nframes)

    # modify the sound here
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
    data = data + add_data_all
    data = data + add_data_left
    data = data + add_data_right

    if movement.all > 0.4:
        data = data + ss.calculate_weighted_segment(old_movement.all,
                                                    new_movement.all,
                                                    nframes,
                                                    sounds["kick_add"][
                                                        "data"])

    data = ss.limit_sound(data)

    old_movement = new_movement
    res_data = data.astype(np.int16).tostring()

    return res_data, pyaudio.paContinue


stream = p.open(format=p.get_format_from_width(base["wave"].getsampwidth()),
                channels=base["wave"].getnchannels(),
                rate=base["wave"].getframerate(),
                output=True,
                stream_callback=callback)
stream.start_stream()

# ----------------------------- init video ----------------------------
config_video = {
    "frame width": 500,
    "min area": 2000,
    "frame area": 0,
    "tracked frames": 40
}

video_channel = 0
vs = VideoStream(src=video_channel).start()
print("Video Channel: " + str(video_channel))
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
