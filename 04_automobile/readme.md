# auto loero - ar lineáris regresszió feladat keras modellellel

## altalnos bevezeto
Adat tisztazasa, heat map vizsgalata.
valojaban szam adatkent is latjuk az ertekeket.
a
```shell
data_frame.corr()
```
a kozel 1 (egyenes aranyossag), es kozel -1 (forditott aranyossag) az erdekes.

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

## csv elokeszitese 
Hozd le a csv file, toltsd be kulso programmal (pl csv), nezd vegig, torold ki azokat a sorokat
amelyek arai `?` ertekuek

# automobile.py
ebben a forrasban kell implementalni a kovetkezo fv-eket.

## replace_question_mark fv
Keszits egy `replace_question_mark` fv-t, ami a parameterben kapott
dataframeben elofordulo `?` jeleket rendre kicsereli valodi ertekre.
Az atlag erket tedd be a helyere, amlyet elotte ki kell szamolnod.

## load_data fv
Keszits egy fv-t, 
1. ami betolti a letoltott csv,  
1. kicserelteti a `?` jeleket szam adatokra, hasznalva a `replace_question_mark` fv-t
1. majd visszaadja a mar hasznalhato dataframe-et

## split_data_train_test
Keszits egy fv-t, ami
1. megkeveri az adat sorokat (df.sample(frac=1))
1. ket parametert kap
    1. az adatokat tartalmayzo dataframe-et
    1. train_size: a tanito es az tesztelo adatok aranyanak szamat
1. visszaadja a kapott dataframe-bol kepzett numpy tomboket 
    1. train_features: az adatsorbol csak az ar `price` oszlop kell, az osszes adatsor 
       kozul train_size * osszes darabszam sort tartalmazza
    1. train_labels: az adatsorbol csak az loero `horsepower` oszlop kell, az osszes adatsor 
       kozul train_size * osszes darabszam sort tartalmazza
    1. test_features: az adatsorbol csak az ar `price` oszlop kell, az osszes adatsor kell, 
       ami nincs benne a train_features tombben
    1. test_labels: az adatsorbol csak az loero `horsepower` oszlop kell, az osszes adatsor kell, 
       ami nincs benne a train_features tombben

## linear_regression fv
Keszits egy `linear_regression` fv-t, ami
1. Letrehoz egy szekvencialis modellt. Hasznald a `tf.keras.Sequential`-t, ami 1 Dense retegbol
all. A reteg `units` es `input_shape` erteke is 1, mert csak a loero - ar viszonyat vizsgaljuk.
1. Forditsd le a modellt a model `compile` metodusaval
    1. hssznald az Adam optimalizalot, tanulasi rata legyen 0.2
    1. a veszteseg fv-nek hasznald a beepitett `mean_absolute_error`  fv-t
1. hangold be a modelled a model `fit` metodusaval
    1. epochs legyen 200
    2. verbose 2
    3. a kapott feature adatokatbol hasznaljon 20% az ellenorzesre
 1. add vissza a modelt, es a historyt

# main.py
A kovetkezo kodot a `main.py`-ba kellene irni, az elozo fv-eket fogjuk hasznalni.
1. betolteti az adatokat a csv-bol a `load_data` fv segitsegeve.
1. `split_data_train_test` segitsegevel letrehozza a szukseges 4 tombot. 
   train_features, train_labels, test_features, test_labels
1. a kapott tombok elemeit konvertald at egeszrol valos ertekre
```python
    tomb = np.asarray(tomb).astype(np.float32)
```
1.hozzunk letre egy modellt, es tanitsuk is be az elozoleg definialt `linear_regression`
   metodus hivasaval. A visszateresi erteket tarold el. 
   Parametrenek hasznald a `split_data_train_test` altal visszaadott test feature, es teszt labels 
   tomboket
1. Mekkora ar tartozik a 140 loerohoz? Ird ki ezt az erteket. Hasznald a model `predict` metodusat
1. Ellenoriztesd a modellunket. Hivd meg a modelunkon az `evaluate` metodust. 
   1. parameternek hasznald a 2 test tombot,
   1. a `verbose` erteke legyen 1 
1. mentsd el a modelt! Hasnzald a modellunk `save` metodusat a 'saved_model/my_model' parameterrel.

Ennyi

Mindenkeppen rajzoltasd ki a modelled  adatiat.
1. diagramban jelenitsuk meg a 
    1. a loerohoz tartozo ar ertekeket 
    ```python
        x = tf.linspace(0.0, 250, 251)
        y = horsepower_model.predict(x)
    ```
    1. az eredeti adatokat
1. a veszteseg ertekeket a history objektumbol ()


Szorgalmi feladat 1

1. Egy `model_loader.py` fileban
1. toltsd be az elozo modell-t hasznald a `tf.keras.models.load_model` fvt, 
1. es ird ki a 111 loerohoz tartozo arat (`model.predict`)

Tensorboard elemzeshez kapcsoljuk be a loggolast.
```python
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
```

```python
    history = horsepower_model.fit(
        ...
        callbacks=tensorboard_callback,
        # Calculate validation results on 20% of the training data
        ...
    )
```

Nezd meg ternsorboard-al a modelled adatait, azoknak az alakulasat.

Melyik tartomanyban teljesit jol a model?


Szorgalmi feladat 2
Szervezd at a modelledet ugy,
hogy 3 reteg legyen
1. marad ugyanaz
1. dense reteg 64 egyseggel, aktivizacios fv-e 'relu'
1. dense reteg 64 egyseggel, aktivizacios fv-e 'relu'
1. dense reteg 1 egyseggel

Melyik tartomanyban teljesit jol a model?
Javult valamit? Melyik tartomany lett jobb, melyik rosszabb?
Ahol rossz miert rossz?
