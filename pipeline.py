import os
import pandas as pd
import numpy as np
import dill
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def cleaning(X):

    import pandas as pd

    # Входные фичи — все колонки вида utm_*, device_*, geo_*,
    # кроме 'device_model':
    input_cols=[
       'utm_source',
       'utm_medium',
       'utm_campaign',
       'utm_adcontent',
       'utm_keyword',
       'device_category',
       'device_brand',
       'device_screen_resolution',
       'device_browser',
       'device_os',
       'geo_country',
       'geo_city'
    ]

    df = X[input_cols]

    # Функции для заполнения пропусков в 'device_brand' и 'device_os':
    def brand_imputer(x):
        x_brand = x['device_brand']
        if pd.isna(x_brand) or x_brand == '(not set)': # Значение '(not set)' тоже считаем пропуском.
            if x['device_os'] in ['iOS', 'Macintosh']:
                return 'Apple_imputed' # Лучше, чтобы заполненые значения отличались от оригинальных.
            return 'other'
        return x_brand
    
    def os_imputer(x):
        x_os = x['device_os']
        x_cat = x['device_category']
        if pd.isna(x_os) or x_os == '(not set)':
            if x['device_brand'] == 'Apple':
                if x_cat != 'desktop':
                    return 'iOS_imputed'
                return 'Macintosh_imputed'
            if x_cat != 'desktop':
                return 'Android_imputed'
            return 'other'
        return x_os

    df['device_brand'] = df.apply(brand_imputer, axis=1)
    df['device_os'] = df.apply(os_imputer, axis=1)
    df = df.fillna('other')
    df = df.replace('(not set)', 'other')
    df = df.replace('(none)', 'other')

    # Преобразуем разрешение экрана в числовые фичи:

    def scr_res(res_str):
        res_list = res_str.split('x')
        if len(res_list) == 2 and res_list[0].isnumeric() and res_list[1].isnumeric():
            return int(res_list[0]), int(res_list[1])
        else:
            return 0, 0

    df['pixel'] = df['device_screen_resolution'].apply(lambda x: scr_res(x)[0] * scr_res(x)[1])
    df['x_pixel'] = df['device_screen_resolution'].apply(lambda x: scr_res(x)[0])
    df['y_pixel'] = df['device_screen_resolution'].apply(lambda x: scr_res(x)[1])

    df = df.drop(['device_screen_resolution'], axis=1)

    # Заменим нулевые значения разрешения на медиану:

    mx = df['x_pixel'].median()
    my = df['y_pixel'].median()
    for i in df.index:
        if df.loc[i, 'pixel'] == 0:
            df.loc[i, 'pixel'] = int(mx*my)
            df.loc[i, 'x_pixel'] = int(mx)
            df.loc[i, 'y_pixel'] = int(my)

    
    return df


def new_features(X):

    import math

    df = X.copy()

    # Пиксели экрана:

    df['log_pixel'] = df['pixel'].apply(lambda x: math.log(x))
    df['pixel'] = df['pixel'].apply(lambda x: min(x, 5 * 10**6))
    df['log_x_pixel'] = df['x_pixel'].apply(lambda x: math.log(x))
    df['x_pixel'] = df['x_pixel'].apply(lambda x: min(x, 2500))
    df['log_y_pixel'] = df['y_pixel'].apply(lambda x: math.log(x))
    df['y_pixel'] = df['y_pixel'].apply(lambda x: min(x, 2000))

    # Города:

    def city_reduce(df):
        x = float(df['gr'])
        if x < 1000:
            return str(int(math.log(x)))
        else:
            return df['geo_city']

    gr = df.groupby('geo_city')[['geo_city']].transform('count')
    df['gr'] = gr['geo_city']
    df['geo_city'] = df.apply(city_reduce, axis=1)
    df = df.drop(['gr'], axis=1)

    # Трафик:

    df['organic_traffic'] = df['utm_medium'].apply(lambda x: int(x in ['organic', 'referral', 'other']))
    df['mobile_traffic'] = df['device_category'].apply(lambda x: int(x == 'mobile'))
    
    return df


def main():
    path_dir = os.path.dirname(__file__)
    path = os.path.join(path_dir, 'data', 'ga_sessions.csv')
    sess = pd.read_csv(path, low_memory=False)

    path = os.path.join(path_dir, 'data', 'ga_hits.csv')
    hits = pd.read_csv(path)[['session_id', 'event_action']]

    #Целевые значения параметра 'event_action':
    target_events =[
       'sub_car_claim_click',
       'sub_car_claim_submit_click',
       'sub_open_dialog_click',
       'sub_custom_question_submit_click',
       'sub_call_number_click',
       'sub_callback_submit_click',
       'sub_submit_success',
       'sub_car_request_submit_click'
    ]

    hits['target'] = hits['event_action'].apply(lambda x: int(x in target_events))

    hits_df = pd.DataFrame(hits.groupby('session_id')['target'].sum()).reset_index()
    hits_df['label'] = hits_df['target'].apply(lambda x: int(x != 0))
    hits_df = hits_df.drop(['target'], axis=1)
    df = pd.merge(sess, hits_df, how='inner', on='session_id')

    target = 'label'
    X = df.drop([target], axis=1)
    y = df[target]
    
    col_transformer = ColumnTransformer(transformers=[
        ('scaler', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('ecoder', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object))
    ])

    model = MLPClassifier(
        activation='logistic', 
        solver='adam', 
        max_iter=500, 
        hidden_layer_sizes=(100,30),
        random_state=42
    )
    pipe = Pipeline(steps=[
        ('cleaning', FunctionTransformer(cleaning)),
        ('new_features', FunctionTransformer(new_features)),
        ('encoding', col_transformer),
        ('model', model)
    ])


    pipe.fit(X, y)

    proba_full = pipe.predict_proba(X)[:, 1]
    auc_full = roc_auc_score(y, proba_full)
    threshold_prob = 0.03

    metadata = {}
    metadata['version'] = '1.3'
    metadata['model'] = f'{type(pipe.named_steps["model"]).__name__}'
    metadata['ROC AUC'] = f'{auc_full:.4f}'
    metadata['threshold'] = f'{threshold_prob}'

    full_model = {}
    full_model['pipe'] = pipe
    full_model['metadata'] = metadata

    dump_path = os.path.join(path_dir, 'dill_pipe.pkl')
    with open(dump_path, 'wb') as file:
        dill.dump(full_model, file)
    print('Пайплайн с моделью сохранён в файл ' + dump_path)

if __name__ == '__main__':
    main()
