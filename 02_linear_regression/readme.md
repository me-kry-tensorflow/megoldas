#Linearis regresszi sajat model osztallyal feladat

## globalis valtozok
TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 1000

## get_data fuggveny
Generalj egy majdnem linearis adathalmazt (`x`, `y`). A fuggveny ezt a ket tombot adja vissza.
globals.py fileban hozz letre egy parameter nelkuli `generate_data` fuggvenyt.
* szuksegunk van egy `x` tombre, amely `NUM_EXAMPLES` elemszamu, ertekei normal eloszlast kovetnek.
* hozz letre egy `noise` tombot, amiben `NUM_EXAMPLES` elemszamu, ertekei normal eloszlast kovetnek
* szuksegunk van egy `y` tombre melyben az elemek az `x` fuggvenye, de hibaval terhelt (`noise`). 
  x * TRUE_W + TRUE_B + noise
* terjunk vissza a ket ertekkel `x`, `y`
* a veletlen tartalmu tombokhoz hasznaljuk a tensorflow random normal metodusat
https://www.tensorflow.org/api_docs/python/tf/random/normal

## alapadatok generalasa, megjelenitese
1. A `main.py` fileban hivd meg az elozo `get_data` fvt. az adatokat (`x`, `y`) tarold el egy egy globalis valtozoban.
2. A kapott ertekeket jelenitsd meg egy diagramban a matplotlib segitsegevel (kek pontok az adatok).
Ez lesz az adathalmaz, amire szeretnenk a modellunket betanitani.
   
## MyModel osztaly letrehozasa
Hozzunk lerte egy uj my-model.py filet, amiben az osztaly definicio lesz.
1. Hozzunk letre egy `MyModel` osztalyt, ami a `tf.Module`-bol szarmazik.
2. Hozzuk letre az `_init__(self, **kwargs)` metodust
3. Hivjuk meg az ososztaly konstruktorat 
```python
super().__init__(**kwargs)
```   
4. Hozzunk letre egy `w` _adattagot_, amely tf.Variable, erteke 10
5. Hozzunk letre egy `b` _adattagot_, amely tf.Variable, erteke 0
6. Definialjuk a `_call__` metodust, aminek van egy `x` parameter. 
Visszaadja a szamolt erteket w*x+b
   

## model letrehozas, kiindulo adatok megjelenitese
1. A `main.py`-ban hozzuk letre a modellt.
2. A modelunk segitsegevel rajzoltassuk ki az alapadatokat, es a szamolt ertekeket mine `x` ertekhez egy diagramban
    a. x,y (kek szinnel)
    b. x model(x) piros szinnel
3. Irjuk ki a kezdeti veszteseg, w, b ertekeket (model.w, model.b, loss(y, model(x))   

## veszteseg fv letrehozasa
Szuksegunk van egy `loss` veszteseg fv-re, amely a valosagos es szamolt eremeny ertekek kulonbsegek gyokeinek atlaga.
globals.py fileban
* hasznaljuk a tf.reduce_mean, es tf.square metodusokat
* parametere a ket ertek: tenyleges, szamolt y
* terjunk vissza a szamolt ertekkel

## training fv
Kiszamolja a tenyleges es a szamolt kimenet ertek kulonbseget, majd korrigalja a `w` es `b` erteket.

## training_loop
1. Cilkusban hivja a training fv. egy elore beallitott ertekszer (`epochs`)
2. gyujts ossze a Ws, Wb ertekeket egy tombbe. Miutan a train valtoztatta a modell-t, olvasd ki az aktualis parametereket,
   es fuzd fel a tomb vegere.
3. add vissza az epochs, Ws, Wb ertekeket

## betanitas, eredmenyek kiirasa
1. hivd meg a `training_loop` fv a megfelelo parameterekkel. 


##
Szukseges ismeretek:
### a tombol b tomb kivonasa
```python
a.assign_sub(b)
```

### a tombhoz b tomb hozza fesulese a vegere
```python
a.append(b)
```

### 10 elemu tomb letrehozasa elemei: 1, 2, 3, ...
```python
tomb = range(10)
```

### Gradiens szamitasa ket ertek fveben
```python
    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        veszteseg = szamitas(tenyleges_y, szamitott_y)

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(veszteseg, [model.w, model.b])
```

### tomb ertekeinek atlaga
```python
tf.reduce_mean(tomb)
```

### y = x * x = x^2
```python
tf.square(x)
```

### x, y fv pontok megjelenitese
```python
    plt.figure()
    plt.scatter(x, y, c="b")
    plt.show()
```

### x, y fv kirajzolas kekkel
```python
    plt.figure()
    plt.plot(x, y, c="b")
    plt.show()
```

### x, y fv kirajzolas piros szaggatott vonallal
```python
    plt.figure()
    plt.plot(np.arange(10), np.arange(10), 'r--')
    plt.show()
```