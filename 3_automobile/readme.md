#lineáris regresszió feladat keras modellel


Adat tisztazasa, heat map vizsgalata.
valojaban szam adatkent is latjuk az ertekeket.
a
```shell
data_frame.corr()
```
a kozel 1, es kozel -1 az erdekes.

Fontos, hogy ha van hiba az oszlopban, akkor nem veszi fel a matrixba.
Ezert toroljuk az excelbol a sorokat, arbol az ar `?` erteku.
Ha kodbol toroljuk, ki, akkor mar nem szamolja ujbol, nem lesz benne az ar, pedig pont az kellene.

```python
data = data.drop(data[data.price == '?'].index)
```
Ahol szam erteknek kellene lennie, de `?`, ott le kell azt cserelni az atlagra.

Altalanos megoldas:
```python
        df_temp = data[data[column_name] != '?']
        normalised_mean = df_temp[column_name].astype(int).mean()
        data[column_name] = data[column_name].replace('?', normalised_mean).astype(int)
```
Van meg valahol `?` ertek? Ha van miel lehetne helyettesiteni?




Tensorflow elemzeshez kapcsoljuk be a loggolast
```python
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

```python
    history = horsepower_model.fit(
        feature, train_labels,
        epochs=200,
        # suppress logging
        verbose=2,
        callbacks=tensorboard_callback,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)
```