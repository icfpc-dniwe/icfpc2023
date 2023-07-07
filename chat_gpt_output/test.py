import json
import math


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def is_valid_placement(musicians, placements, x, y):
    for placement in placements:
        if placement is not None and calculate_distance(x, y, placement['position']['x'], placement['position']['y']) <= 10:
            return False
    return True


def calculate_happiness(attendee, musicians, placements):
    happiness = 0
    for i, taste in enumerate(attendee['tastes']):
        for j, musician in enumerate(musicians):
            if musician == i:
                distance = calculate_distance(attendee['position']['x'], attendee['position']['y'], placements[j]['position']['x'], placements[j]['position']['y'])
                happiness += taste / (distance + 1)
    return happiness


def find_best_placement(attendee, musicians, placements):
    max_happiness = 0
    best_placement = None

    for x in range(len(placements)):
        for y in range(len(placements[x])):
            if is_valid_placement(musicians, placements, x, y):
                placements[x][y] = attendee
                happiness = calculate_happiness(attendee, musicians, placements)
                if happiness > max_happiness:
                    max_happiness = happiness
                    best_placement = (x, y)
                placements[x][y] = None

    return best_placement


def solve_problem(input_data):
    room_width = int(input_data['room_width'])
    room_height = int(input_data['room_height'])
    stage_width = int(input_data['stage_width'])
    stage_height = int(input_data['stage_height'])
    stage_bottom_left = input_data['stage_bottom_left']
    musicians = input_data['musicians']
    attendees = input_data['attendees']

    num_musicians = len(musicians)
    placements = [[None] * stage_height for _ in range(stage_width)]

    for attendee in attendees:
        best_placement = find_best_placement(attendee, musicians, placements)
        if best_placement:
            x, y = best_placement
            placements[x][y] = {'position': {'x': attendee['x'], 'y': attendee['y']}}

    return {'placements': placements}


# Example input
input_data = {
    "room_width": 2000.0,
    "room_height": 5000.0,
    "stage_width": 1000.0,
    "stage_height": 2000.0,
    "stage_bottom_left": [500.0, 0.0],
    "musicians": [0, 1, 0],
    "attendees": [
        {"x": 1000.0, "y": 5000.0, "tastes": [1000.0, -1000.0]},
        {"x": 2000.0, "y": 1000.0, "tastes": [2000.0, 2000.0]},
        {"x": 1100.0, "y": 8000.0, "tastes": [8000.0, 15000.0]}
    ]
}

solution = solve_problem(input_data)
print(json.dumps(solution))
