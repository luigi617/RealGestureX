# gesture_commands.py

def map_gesture_to_command(gesture):
    """
    Maps a recognized gesture to a specific command.

    Args:
        gesture (str): The recognized gesture label.

    Returns:
        str: The corresponding command.
    """
    gesture_command_mapping = {
        # Static Gestures
        'pointing': 'select_item',
        'open_palm': 'open_menu',
        'thumb_index_touch': 'zoom_in',
        'thumb_middle_touch': 'zoom_out',
        'fist': 'pause',

        # Dynamic Gestures
        'swipe_up': 'scroll_up',
        'swipe_down': 'scroll_down',
        'swipe_left': 'prev_slide',
        'swipe_right': 'next_slide'
    }

    return gesture_command_mapping.get(gesture, 'no_action')
