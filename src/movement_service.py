from collections import namedtuple
import cv2
import imutils

Movement = namedtuple('Movement', ['all', 'left', 'right'])


def process_new_movement(movement, old_movement):
    if movement.all < old_movement.all * 0.9:
        new_movement_all = old_movement.all * 0.9
    else:
        new_movement_all = movement.all

    if movement.left < old_movement.left * 0.9:
        new_movement_left = old_movement.left * 0.9
    else:
        new_movement_left = movement.left

    if movement.right < old_movement.right * 0.9:
        new_movement_right = old_movement.right * 0.9
    else:
        new_movement_right = movement.right

    return Movement(new_movement_all, new_movement_left, new_movement_right)


def calculate_movement(frames, config):
    # compute the absolute difference between the current frame and
    # first frame
    frame_delta = cv2.absdiff(frames.popleft(), frames[-1])
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    frame_delta_left = frame_delta[:, 0:(config["frame width"] // 2)].copy()
    frame_delta_right = frame_delta[:, (config["frame width"] // 2):config[
        "frame width"]].copy()

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh_left = thresh[:, 0:(config["frame width"] // 2)].copy()
    thresh_right = thresh[:,
                   (config["frame width"] // 2):config["frame width"]].copy()

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
            if contour_area < config["min area"]:
                continue

            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.drawContours(frame, c, -1, (0, 255, 0), 2)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = area + contour_area

        movements[i] = round(area / config["frame area"], 4)

    movement = Movement(sum(movements), 2 * movements[0], 2 * movements[1])

    return movement, frame_delta_left, frame_delta_right
