# Klasifikacija emotikona uporabom konvolucijske neuronske mreže

## Upute za rad sa programom

Za pokretanje je potrebno napraviti python virtualenv (git bash):

```bash
virtualenv venv -p /c/python36/python.exe
. venv/Scripts/activate
export PYTHONIOENCDOING=utf8
```

Te zatim instalirati potrebne biblioteke:

```bash

pip install -r requirements.txt
```

**Napomena**:

Ukoliko računalo ne podržava CUDA GPU, umjesto tensorflow-gpu dovoljno je
instalirati tensorflow (npr. zamjenom zapisa u requirements.txt i pokretanjem
`pip install -r requirements.txt`).

## Treniranje

Treniranje je kumulativno pa za kretanje ispočetka je potrebno pozvati
`rm -rf .model/* logdir/*`.

```bash
python -u train.py
```

U drugom terminalu je moguće pratiti detaljte pokretanjem:

```bash
tensorboard --logdir ./logdir
```

## Predviđanje (klasificiranje na odabranim primjerima)

```bash
python -u predict.py
```
