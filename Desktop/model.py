import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('preprocessed_data.csv')

    x = data.drop(['Survived'], axis=1)
    y = data.Survived

    loss_function = 'Logloss'
    learning_rate = None
    iterations = 1000
    custom_loss = 'Accuracy'

    model = CatBoostClassifier(iterations=iterations,
                               custom_loss=[custom_loss],
                               loss_function=loss_function,
                               learning_rate=learning_rate
                               )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=.85,
        random_state=42
    )
    is_categorical_feature = np.where(x_train.dtypes != float)[0]

    model.fit(x_train,
              y_train,
              cat_features=is_categorical_feature,
              eval_set=(x_test, y_test)
              )

    cv(Pool(x,
            y,
            cat_features=is_categorical_feature
            ),
       model.get_params(),
       fold_count=5
       )

    model.save_model('trained_model.cbm')
