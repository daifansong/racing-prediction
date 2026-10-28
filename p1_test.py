import importlib
import os
import re
import sys
import unicodedata

import pandas as pd

from utils_test import predict_score, read_x_features, to_matrix

P1_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = '/Users/daifansong/Nature/projects/racing/data/feature-v1'
OUT_DIR = os.path.join(P1_DIR, 'out', 'ml')


MODEL_NAME = 'lightgbm'
RACE_ID = '2004-09-15_HV'


SHOW_COLUMNS = (
    'race_id', 'date', 'venue', 'race_no',
    'HorseNo', 'horse_name', 'jockey', 'trainer',
    'Track', 'Course', 'track_condition',
    'place_num', 'distance', 'field_size',
    'y_win', 'y_place',
)
TABLE_HEADERS = (
    'HorseNo', 'horse_name', 'jockey', 'trainer', 'odd',
    'pred_win', 'pred_place', 'place', 'y_win', 'y_place', 'win_ok', 'place_ok',
)
RACE_ID_PATTERN = re.compile(r'^(\d{4})-(\d{2})-(\d{2})_(ST|HV)$')
TASK_NAMES = ('win_with_odd', 'place_with_odd')

def meeting_id(race_id):
    match = RACE_ID_PATTERN.fullmatch(race_id)
    if match is None:
        raise ValueError(f'\t{race_id} saved failed')
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f'\t{race_id} saved failed')
    return race_id, year, month

def season_year(year, month):
    if month >= 9:
        return year
    return year - 1

def list_race_files(day_dir, query):
    if not os.path.isdir(day_dir):
        raise ValueError(f'\t{query} saved failed')
    names = [
        name for name in os.listdir(day_dir)
        if name.startswith('race_') and name.endswith('.csv')
    ]
    names.sort()
    if not names:
        raise ValueError(f'\t{query} saved failed')
    return [os.path.join(day_dir, name) for name in names]

def resolve_model(model_name, p1_dir, out_dir):
    selected_path = os.path.join(p1_dir, 'models', 'selected', model_name + '.py')
    all_path = os.path.join(p1_dir, 'models', 'all', model_name + '.py')
    if os.path.isfile(selected_path):
        return 'models.selected', os.path.join(out_dir, 'selected')
    if os.path.isfile(all_path):
        return 'models.all', os.path.join(out_dir, 'all')
    raise ValueError(f'\t{model_name} saved failed')

def load_task_models(model_name, package_name, model_root):
    model_cls = importlib.import_module(f'{package_name}.{model_name}').Model
    model_dir = os.path.join(model_root, model_name)
    loaded = {}
    for task_name in TASK_NAMES:
        model_path = os.path.join(model_dir, task_name)
        if not os.path.isfile(model_path + '.txt'):
            raise ValueError(f'\t{task_name} saved failed')
        model = model_cls()
        model.load(model_path)
        loaded[task_name] = model
    return loaded

def one_value(frame, column, race_id):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1 or values[0] == '':
        raise ValueError(f'\t{race_id} saved failed')
    return values[0]

def load_race(path, feature_names, expect_race_id):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    label = expect_race_id if expect_race_id is not None else os.path.basename(path)
    if len(frame) == 0 or 'race_id' not in frame.columns:
        raise ValueError(f'\t{label} saved failed')
    race_id = one_value(frame, 'race_id', label)
    if expect_race_id is not None and race_id != expect_race_id:
        raise ValueError(f'\t{expect_race_id} saved failed')
    missing = [name for name in (*feature_names, *SHOW_COLUMNS) if name not in frame.columns]
    if missing:
        raise ValueError(f'\t{race_id} saved failed')
    if one_value(frame, 'field_size', race_id) != str(len(frame)):
        raise ValueError(f'\t{race_id} saved failed')
    return frame, race_id

def binary_column(frame, column, race_id):
    values = []
    for text in frame[column].tolist():
        if text not in ('0', '1'):
            raise ValueError(f'\t{race_id} saved failed')
        values.append(int(text))
    return values

def horse_numbers(frame, race_id):
    numbers = []
    for text in frame['HorseNo'].tolist():
        if not text.isdigit():
            raise ValueError(f'\t{race_id} saved failed')
        numbers.append(int(text))
    if len(set(numbers)) != len(numbers):
        raise ValueError(f'\t{race_id} saved failed')
    return numbers

def mark(ok):
    if ok:
        return '✅'
    return '❌'

def pick_mark(predicted, actual):
    if predicted and actual:
        return '✅'
    if predicted:
        return '❌'
    if actual:
        return '🔵'
    return ''

def choose_win(scores, horse_nos):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), horse_nos[i]))
    best = order[0]
    tied = [i for i in order if scores[i] == scores[best]]
    return best, tied

def choose_place(scores, horse_nos):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), horse_nos[i]))
    chosen_count = min(3, len(order))
    chosen = order[:chosen_count]
    cutoff = scores[chosen[-1]]
    at_cutoff = [i for i in order if scores[i] == cutoff]
    if any(index not in set(chosen) for index in at_cutoff):
        return chosen, at_cutoff
    return chosen, []

def number_list(indices, horse_nos):
    ordered = sorted(indices, key=lambda i: horse_nos[i])
    return ','.join(str(horse_nos[i]) for i in ordered)

def display_width(text):
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width

def format_table(headers, rows):
    widths = [display_width(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], display_width(cell))

    def line(cells):
        padded = []
        for index, cell in enumerate(cells):
            gap = widths[index] - display_width(cell)
            padded.append(cell + (' ' * gap))
        return '  '.join(padded)

    return '\n'.join([line(headers), *(line(row) for row in rows)])

def print_race(frame, race_id, win_score, place_score):
    horse_nos = horse_numbers(frame, race_id)
    y_win = binary_column(frame, 'y_win', race_id)
    y_place = binary_column(frame, 'y_place', race_id)
    names = frame['horse_name'].tolist()
    win_index, win_tied = choose_win(win_score, horse_nos)
    place_indices, place_border = choose_place(place_score, horse_nos)
    place_set = set(place_indices)
    pred_win = [1 if index == win_index else 0 for index in range(len(frame))]
    pred_place = [1 if index in place_set else 0 for index in range(len(frame))]
    rows = []
    for index in sorted(range(len(frame)), key=lambda i: horse_nos[i]):
        rows.append((
            str(horse_nos[index]),
            names[index],
            frame['jockey'].iloc[index],
            frame['trainer'].iloc[index],
            frame['win_odd'].iloc[index],
            f'{float(win_score[index]):.4f}',
            f'{float(place_score[index]):.4f}',
            frame['place_num'].iloc[index],
            str(y_win[index]),
            str(y_place[index]),
            pick_mark(pred_win[index] == 1, y_win[index] == 1),
            pick_mark(pred_place[index] == 1, y_place[index] == 1),
        ))
    date = one_value(frame, 'date', race_id)
    venue = one_value(frame, 'venue', race_id)
    race_no = int(one_value(frame, 'race_no', race_id))
    distance = one_value(frame, 'distance', race_id)
    track = one_value(frame, 'Track', race_id)
    course = one_value(frame, 'Course', race_id)
    going = one_value(frame, 'track_condition', race_id)
    field_size = one_value(frame, 'field_size', race_id)
    print(
        f'{race_id}  {date}  {venue}  R{race_no:02d}  {distance}  {track}  {course}  {going}  field {field_size}',
        flush=True,
    )
    print(flush=True)
    print(format_table(TABLE_HEADERS, rows), flush=True)
    print(flush=True)
    actual_win = ', '.join(
        f'{horse_nos[index]} {names[index]}'
        for index in sorted(range(len(frame)), key=lambda i: horse_nos[i])
        if y_win[index] == 1
    )
    win_tie = ''
    if len(win_tied) > 1:
        win_tie = f'  tie {number_list(win_tied, horse_nos)}'
    race_win_ok = y_win[win_index] == 1
    print(
        f'win pick {horse_nos[win_index]} {names[win_index]}{win_tie}  actual {actual_win}  {mark(race_win_ok)}',
        flush=True,
    )
    actual_place = number_list([index for index in range(len(frame)) if y_place[index] == 1], horse_nos)
    place_tie = ''
    if place_border:
        place_tie = f'  tie {number_list(place_border, horse_nos)}'
    race_place_ok = {horse_nos[index] for index in place_indices} == {
        horse_nos[index] for index in range(len(frame)) if y_place[index] == 1
    }
    print(
        f'place pick {number_list(place_indices, horse_nos)}{place_tie}  actual {actual_place}  {mark(race_place_ok)}',
        flush=True,
    )
    print(flush=True)
    return race_win_ok, race_place_ok

def main():
    if not MODEL_NAME.isidentifier():
        raise ValueError(f'\t{MODEL_NAME} saved failed')
    day_id, year, month = meeting_id(RACE_ID)
    day_dir = os.path.join(FEATURE_DIR, str(season_year(year, month)), day_id)
    paths = list_race_files(day_dir, day_id)
    feature_names = read_x_features()
    package_name, model_root = resolve_model(MODEL_NAME, P1_DIR, OUT_DIR)
    models = load_task_models(MODEL_NAME, package_name, model_root)
    win_hits = 0
    place_hits = 0
    for path in paths:
        frame, race_id = load_race(path, feature_names, None)
        matrix = to_matrix(frame, feature_names)
        win_score = predict_score(models['win_with_odd'], matrix, len(frame), 'win_with_odd')
        place_score = predict_score(models['place_with_odd'], matrix, len(frame), 'place_with_odd')
        win_ok, place_ok = print_race(frame, race_id, win_score, place_score)
        win_hits += int(win_ok)
        place_hits += int(place_ok)
    count = len(paths)
    print(f'day {day_id}  races {count}  win {win_hits}/{count}  place {place_hits}/{count}', flush=True)

if __name__ == '__main__':
    main()
