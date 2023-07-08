from pathlib import Path
import json
from src.mytypes import ProblemInfo, ProblemSolution, Room, Stage, Placement, Attendee, Pillar


def read_problem(json_path: Path) -> ProblemInfo:
    with json_path.open('r') as f:
        info = json.load(f)
    return ProblemInfo(
        room=Room(width=info['room_width'], height=info['room_height']),
        stage=Stage(width=info['stage_width'], height=info['stage_height'],
                    bottom_x=info['stage_bottom_left'][0], bottom_y=info['stage_bottom_left'][1]),
        musicians=info['musicians'],
        attendees=[Attendee(x=cur_att['x'], y=cur_att['y'], tastes=cur_att['tastes'])
                   for cur_att in info['attendees']],
        pillars=[Pillar(x=cur_p['center'][0], y=cur_p['center'][1], radius=cur_p['radius'])
                 for cur_p in info['pillars']]
    )


def load_solution(json_path: Path) -> ProblemSolution:
    with json_path.open('r') as f:
        placements = json.load(f)
    return ProblemSolution(placements=[Placement(x=p['x'], y=p['y']) for p in placements['placements']])


def save_solution(solution: ProblemSolution, save_path: Path) -> None:
    dict_out = {
        'placements': [
            {'x': cur_placement.x, 'y': cur_placement.y} for cur_placement in solution.placements
        ]
    }
    with save_path.open('w') as f:
        json.dump(dict_out, f)
