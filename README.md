# Aplikace pro automatické prahování obrazu

## Popis projektu
Tato aplikace implementuje tři metody automatického prahování:
1. **Klasická Otsuova metoda** - globální prahování vhodné pro bimodální histogramy
2. **Recursive Otsu** - rozšíření Otsuovy metody pro historické dokumenty (dle [Nina et al. 2010](#))
3. **Sauvolova metoda** - adaptivní lokální prahování

Aplikace nabízí grafické rozhraní pro interaktivní experimentování s různými typy obrazových dat.

## Instalace
1. Vyžaduje Python 3.8+
2. Instalace závislostí:
```bash
pip install opencv-python numpy pillow matplotlib scikit-image
