# 132 列的原子分组。含赔率时 HORSE 多 win_odd，无赔率时解析阶段跳过它。

RACE = [
    'distance', 'width', 'straight', 'is_awt', 'is_hv', 'condition',
    'climate_sin', 'climate_cos', 'is_wet', 'season_sin', 'season_cos',
    'season_early', 'season_mid', 'season_late', 'field_size',
]

HORSE = [
    'draw', 'win_odd', 'jw_c', 'h_rating', 'h_rating_c', 'h_rating_mean',
    'h_rating_last', 'h_rating_delta', 'h_starts', 'h_n', 'h_win', 'h_place',
    'h_starts_prev', 'h_win_prev', 'h_place_prev', 'h_starts_season',
    'h_n_season', 'h_win_season', 'h_place_season', 'h_form_season',
    'h_win_c', 'h_form3', 'h_last_place', 'h_is_debut', 'h_win_dist',
    'h_place_dist', 'h_win_band', 'h_place_band', 'h_win_venue',
    'h_place_venue', 'h_win_surf', 'h_place_surf', 'h_win_rail',
    'h_place_rail', 'h_win5', 'h_place5', 'h_win5_gap', 'h_form5',
    'h_rating_last_win', 'h_jw_last_win', 'h_rating_last_place',
    'h_jw_last_place', 'h_rating_vs_last_win', 'h_jw_vs_last_win',
    'rest_days', 'days_last_win', 'days_last_place', 'h_never_win',
    'h_never_place', 'dist_delta', 'rail_delta', 'condition_delta',
    'jw_vs_last', 'first_this_dist', 'first_this_band', 'venue_change',
    'surf_change', 'first_this_surf', 'jockey_change',
]

JOCKEY = [
    'j_n', 'j_starts', 'j_win', 'j_place', 'j_win_c', 'j_win_venue',
    'j_place_venue', 'j_win_surf', 'j_place_surf', 'j_win_band',
    'j_place_band', 'j_win10', 'j_place10', 'j_win10_gap', 'j_win_s30',
    'j_place_s30', 'j_win_s50', 'j_place_s50', 'j_win_s100', 'j_place_s100',
    'j_win_prev', 'j_place_prev', 'j_starts_prev',
]

TRAINER = [
    't_n', 't_starts', 't_win', 't_place', 't_win_c', 't_win_venue',
    't_place_venue', 't_win_surf', 't_place_surf', 't_win_band',
    't_place_band', 't_win10', 't_place10', 't_win10_gap', 't_win_s30',
    't_place_s30', 't_win_s50', 't_place_s50', 't_win_s100', 't_place_s100',
    't_win_prev', 't_place_prev', 't_starts_prev',
]

HJ = [
    'hj_starts', 'hj_win', 'hj_place',
]

JT = [
    'jt_starts_season', 'jt_win_season', 'jt_place_season',
    'jt_starts_prev', 'jt_win_prev', 'jt_place_prev',
    'jt_starts', 'jt_win', 'jt_place',
]

# None：不切列，整表进一条塔。其余每项是一塔的列名，顺序即 stem 的输入顺序。
LAYOUTS = {
    1: None,
    2: [RACE + HORSE, HJ + JOCKEY + TRAINER + JT],
    3: [RACE, HORSE, HJ + JOCKEY + TRAINER + JT],
    4: [RACE, HORSE, HJ + JOCKEY, TRAINER + JT],
    5: [RACE, HORSE, JOCKEY, TRAINER, HJ + JT],
}
