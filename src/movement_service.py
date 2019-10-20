from collections import namedtuple

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
