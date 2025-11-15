import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap


def plot_varianta(valori_proprii_sortate: np.ndarray, procent_minimal=80, scal=True):
    """
    Argumente:
        valori_proprii_sortate (np.ndarray): Vectorul valorilor proprii (fost 'alpha').
        procent_minimal (int): Pragul procentual pentru criteriul acoperirii minimale.
        scal (bool): Daca este True, se aplica Criteriul Kaiser (varianta > 1).

    Returneaza:
        tuple: Numarul de componente sugerat de (k1: Kaiser, k2: Acoperire, k3: Cattell).
               k1 sau k3 pot fi None, conform logicii originale.
    """
    nr_componente = len(valori_proprii_sortate)

    axe_x_componente = np.arange(1, nr_componente + 1)

    figura = plt.figure(figsize=(8, 5))
    axa_grafic = figura.add_subplot(1, 1, 1)

    axa_grafic.set_title(
        "Plot varianta componente", fontdict={"color": "b", "fontsize": 16}
    )
    axa_grafic.plot(axe_x_componente, valori_proprii_sortate)
    axa_grafic.set_xlabel("Componente", fontsize=12)
    axa_grafic.set_ylabel("Varianta", fontsize=12)

    # Seteaza marcajele pe axa X sa fie 1, 2, 3...
    axa_grafic.set_xticks(axe_x_componente)

    # Adauga puncte rosii pe linia graficului
    axa_grafic.scatter(axe_x_componente, valori_proprii_sortate, c="r")

    kaiser_componente = None
    if scal:
        axa_grafic.axhline(1, c="g", label="Criteriul Kaiser")
        valori_peste_1 = np.where(valori_proprii_sortate > 1)
        kaiser_componente = len(valori_peste_1[0])

    procent_cumulat = np.cumsum(
        valori_proprii_sortate * 100 / np.sum(valori_proprii_sortate)
    )

    # Gaseste primul index unde procentul cumulat depaseste pragul minimal
    # [0][0] selecteaza primul index care indeplineste conditia
    index_acoperire = np.where(procent_cumulat > procent_minimal)[0][0]
    # +1 pentru ca indexul incepe de la 0 (fost 'k2')
    acoperire_componente = index_acoperire + 1

    # Deseneaza linia orizontala corespunzatoare
    eticheta_acoperire = "Acoperire minimala (" + str(procent_minimal) + "%)"
    axa_grafic.axhline(
        valori_proprii_sortate[index_acoperire], c="m", label=eticheta_acoperire
    )

    # --- Criteriul Cattell (Scree Plot / Cot) (k3) ---
    # Calculeaza diferenta de ordinul 1 (panta) (fost 'eps')
    diferenta_1 = (
        valori_proprii_sortate[: nr_componente - 1] - valori_proprii_sortate[1:]
    )
    # Calculeaza diferenta de ordinul 2 (acceleratia/schimbarea pantei) (fost 'sigma')
    diferenta_2 = diferenta_1[: nr_componente - 2] - diferenta_1[1:]

    cattell_componente = None
    # Verifica daca exista vreo valoare negativa in diferenta de ordinul 2
    # O valoare negativa inseamna ca panta a inceput sa creasca (punct de inflexiune / "cot")
    exista_negative = any(diferenta_2 < 0)

    if exista_negative:
        # Gaseste primul punct de inflexiune ("cotul")
        index_cot = np.where(diferenta_2 < 0)[0][0]
        # Numarul de componente este indexul + 2 (logica originala, fost 'k3')
        cattell_componente = index_cot + 2
        # Deseneaza linia orizontala
        axa_grafic.axhline(
            valori_proprii_sortate[cattell_componente - 1],
            c="c",
            label="Criteriul Cattell",
        )

    # Afiseaza legenda si salveaza figura
    axa_grafic.legend()
    plt.savefig("graphics/PlotVarianta.png")

    return kaiser_componente, acoperire_componente, cattell_componente


def show():
    plt.show()


def corelograma(
    dataframe_date: pd.DataFrame,
    titlu="Corelograma",
    vmin=-1,
    cmap="RdYlBu",
    annot=True,
):
    # Initializarea figurii si axei Matplotlib
    figura = plt.figure(figsize=(9, 8))
    axa_grafic = figura.add_subplot(1, 1, 1)

    # Seteaza titlul
    axa_grafic.set_title(titlu, fontdict={"color": "b", "fontsize": 16})

    # Genereaza heatmap-ul folosind Seaborn
    heatmap(dataframe_date, vmin=vmin, vmax=1, cmap=cmap, annot=annot, ax=axa_grafic)

    # Salveaza figura (numele fisierului este titlul)
    plt.savefig("graphics/" + titlu + ".png")


def plot_scoruri_corelatii(
    dataframe_date: pd.DataFrame, var_x="C1", var_y="C2", titlu="Plot scoruri"
):
    """
    Genereaza un grafic scatter plot pentru doua variabile (de obicei scoruri ACP).
    """
    # Initializarea figurii si axei Matplotlib
    figura = plt.figure(figsize=(10, 6))
    axa_grafic = figura.add_subplot(1, 1, 1)

    axa_grafic.set_title(titlu, fontdict={"color": "b", "fontsize": 16})

    axa_grafic.scatter(dataframe_date[var_x], dataframe_date[var_y], c="r")

    plt.savefig("graphics/" + titlu + ".png")
