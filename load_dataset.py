from pathlib import Path
import pandas as pd
import numpy as np

RARE_CLASSES = ['Repair', 'D43', 'D01', 'D11', 'Block crack', 'D0w0']

def _group_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df['class'].isin(RARE_CLASSES), 'class'] = 'Other'
    return df

def _drop_invalid_boxes(df: pd.DataFrame) -> pd.DataFrame:
    invalid = (df['xmin'] >= df['xmax']) | (df['ymin'] >= df['ymax'])
    return df[~invalid].copy()

def load_metadata(train_csv_path: str):
    train_df = pd.read_csv(train_csv_path)
    train_df = _group_rare_classes(train_df)
    train_df = _drop_invalid_boxes(train_df)

    valid_df_grouped = train_df.dropna(subset=['class'])
    class_names = valid_df_grouped['class'].unique()
    class_names = np.insert(class_names, 0, '__background__')
    class_to_int = {label: i for i, label in enumerate(class_names)}
    int_to_class = {i: label for i, label in enumerate(class_names)}

    class_color_map = {
        '__background__': 'black',
        'D00': 'red',
        'D10': 'orange',
        'D20': 'yellow',
        'D40': 'blue',
        'D44': 'purple',
        'D50': 'green',
        'Other': 'cyan'
    }
    category_colors = {class_to_int[name]: color for name, color in class_color_map.items() if name in class_to_int}

    return train_df, class_to_int, int_to_class, category_colors
