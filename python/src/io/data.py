from pathlib import Path
import json
from src.mytypes import ProblemInfo, ProblemSolution, Room, Stage, Placement, Attendee


def read_problem(json_path: Path) -> ProblemInfo:
    with json_path.open('r') as f:
        info = json.load(f)
    return ProblemInfo(
        room=Room(width=info['room_width'], height=info['room_height']),
        stage=Stage(width=info['stage_width'], height=info['stage_height'],
                    bottom_x=info['stage_bottom_left'][0], bottom_y=info['stage_bottom_left'][1]),
        musicians=info['musicians'],
        attendees=[Attendee(x=cur_att['x'], y=cur_att['y'], tastes=cur_att['tastes'])
                   for cur_att in info['attendees']]
    )


def save_solution(solution: ProblemSolution, save_path: Path) -> None:
    dict_out = {
        'placements': [
            {'x': cur_placement['x'], 'y': cur_placement['y']} for cur_placement in solution.placements
        ]
    }
    with save_path.open('w') as f:
        json.dump(dict_out)
