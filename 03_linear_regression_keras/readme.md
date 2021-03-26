## get_data fuggveny
Keszits egy `get_data` fv-t, ami visszaad ket tombot. 
1. A tombok merete legyen mindkettonek azonos, kb 10 meretu, 
   tartalma olyan legyen ami linearis jellegu.
2. Jelenitsd meg az adatokat, hogy ellenorizd jok-e.

## get_model fv
1. Keszits egy linearis halot 1 reteggel (`model`), amiben surun kapcsolodo (Dense) reteg talalhato
https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
Ebben lehet tobb reteg is, de most csak 1 lesz.
2. hivd meg a `model` `compile` metodusat, valasz az 
   - `sgd` optimalizalot, es a 
   - `mean_squared_error` veszteseg fv-t.
3. tanitsd be a halot a `fit` metodus segitsegevel (`model`), ugy, hogy 
    - 500 iteracioban korrigaljon,
    - 20%-at hasznala az adatoknak ellenorzesre
4. a fit metodusnak lesz egy history visszateresi erteke. Ennek a segitsegevel rajzold ki a
vesztesegek, es a validalo ertekek veszteseg ertekeit 
   (plot `history.history['loss']`, `history.history['val_loss']`). 
5. add vissza a `model` objektumot

## main metodus
1. hozd le az adatokat `get_data`
2. a `get_data` visszateresi ertekevel hivd meg a `get_model` metodust, a visszateresi erteket tarold el.
3. kerj egy becslest a `model.predict` metodus hivasaval. tombot kell adni a parameter
ertekenek pl.: [14]
4. Ird ki az erteket
5. Rajzold ki az 1..20 tombre kapott becsult ertekeket.

### beepitett modell hasznalatanak altalanos folyamata
```python
    model = tf.keras.Sequential([
        # retegek felsorolasa
])
    model.compile(optimizer=optimizer_neve, loss=beepitett_loss_fx_neve)
    history = model.fit(xs, ys, epochs=futasi_szam, validation_split=validacio_rata)
```


### tomb letrehozas numpy segitsegevel
```python
np.array([-1.0,  2.0])
```

