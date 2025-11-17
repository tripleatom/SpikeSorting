# sorting_files.json format

## option 1: global parameters

```json
{
    "sorter_name": "mountainsort5",
    "sorter_params": {
        "scheme": "2",
        "detect_threshold": 5.5,
        "detect_sign": 0,
        "detect_time_radius_msec": 0.5,
        "snippet_T1": 20,
        "snippet_T2": 20,
        "npca_per_channel": 3,
        "npca_per_subdivision": 10,
        "snippet_mask_radius": 250,
        "scheme1_detect_channel_radius": 150,
        "scheme2_phase1_detect_channel_radius": 200,
        "scheme2_detect_channel_radius": 120,
        "scheme2_max_num_snippets_per_training_batch": 200,
        "scheme2_training_duration_sec": 300,
        "scheme2_training_recording_sampling_mode": "uniform"
    },
    "recordings": [
        {
            "path": "\\\\10.129.151.108\\xieluanlabs\\xl_cl\\ephys\\sleep\\CnL42SG\\CnL42SG_20251115_133046.rec",
            "shanks": [0, 1, 2, 3],
            "animal_id": "CnL42SG"
        },
        {
            "path": "\\\\10.129.151.108\\xieluanlabs\\xl_cl\\ephys\\sleep\\CnL39SG\\CnL39SG_20251102_210043.rec",
            "shanks": [0, 1, 2, 3],
            "animal_id": "CnL39SG"
        }
    ]
}
```

## option 2: per-recording parameters

```json
{
    "sorter_name": "mountainsort5",
    "recordings": [
        {
            "path": "\\\\10.129.151.108\\xieluanlabs\\xl_cl\\ephys\\sleep\\CnL42SG\\CnL42SG_20251115_133046.rec",
            "shanks": [0, 1, 2, 3],
            "animal_id": "CnL42SG",
            "sorter_params": {
                "scheme": "2",
                "detect_threshold": 5.5,
                "detect_sign": 0,
                "detect_time_radius_msec": 0.5,
                "snippet_T1": 20,
                "snippet_T2": 20,
                "npca_per_channel": 3,
                "npca_per_subdivision": 10,
                "snippet_mask_radius": 250,
                "scheme2_phase1_detect_channel_radius": 200,
                "scheme2_detect_channel_radius": 120,
                "scheme2_max_num_snippets_per_training_batch": 200,
                "scheme2_training_duration_sec": 300,
                "scheme2_training_recording_sampling_mode": "uniform"
            }
        },
        {
            "path": "\\\\10.129.151.108\\xieluanlabs\\xl_cl\\ephys\\sleep\\CnL39SG\\CnL39SG_20251102_210043.rec",
            "shanks": [0],
            "animal_id": "CnL39SG",
            "sorter_params": {
                "scheme": "1",
                "detect_threshold": 4.5,
                "detect_sign": -1
            }
        }
    ]
}
```
