"""The 26 classes, in the index order used for training (torchvision ImageFolder = sorted folder names).

D1 = Rice Field Weed BD v4 (Bangladesh, rice), D2 = MH-Weed16 Individual Weed Species (India, soybean).
Commelina benghalensis occurs in both datasets and is merged into one class.
"""

# (folder name used in training, scientific name, source dataset)
CLASSES = [
    ("1..Lavhala_(Cyperus_Rotundus)", "Cyperus rotundus", "D2"),
    ("10.Gajar_gavat_(Parthenium hysterophorus)", "Parthenium hysterophorus", "D2"),
    ("11.Graceful_Sandmart_(Euphorbia hypericifolia)", "Euphorbia hypericifolia", "D2"),
    ("12.Sicklepod_(Senna obtusifolia)", "Senna obtusifolia", "D2"),
    ("13.Harali_(Cynodon_dactylon)", "Cynodon dactylon", "D2"),
    ("14.Dwarf_cassia_(Chamaecrista pumila)", "Chamaecrista pumila", "D2"),
    ("15.Punarnava _(Boerhaavia diffusa)", "Boerhaavia diffusa", "D2"),
    ("2.Lamber_Quarter_plant(Chenopodium )", "Chenopodium sp.", "D2"),
    ("3.Little_Mallow(Malva parviflora)", "Malva parviflora", "D2"),
    ("4.Moti_dudhi(Euphorbia_geneculata_L)", "Euphorbia geniculata", "D2"),
    ("5.Obscure_morning _glory(Ipomoea obscura)", "Ipomoea obscura", "D2"),
    ("6.Asian_Pigeonwings_(Clitoria Ternatea)", "Clitoria ternatea", "D2"),
    ("7.Bilayat_(Mexicana_Argemone)", "Argemone mexicana", "D2"),
    ("8.Choti_dudhi_(Euphorbia_hirta)", "Euphorbia hirta", "D2"),
    ("9.Digitaria_SP_(Digitaria Sanguinalis )", "Digitaria sanguinalis", "D2"),
    ("Commelina_benghalensis_(Kena)", "Commelina benghalensis", "D1+D2"),
    ("W_CL_01_Alternanthera philoxeroide", "Alternanthera philoxeroides", "D1"),
    ("W_CL_02_Centella asiatica", "Centella asiatica", "D1"),
    ("W_CL_04_Cyperus ochraceus", "Cyperus ochraceus", "D1"),
    ("W_CL_05_Fimbristylis littoralis", "Fimbristylis littoralis", "D1"),
    ("W_CL_06_Ipomoea aquatic", "Ipomoea aquatica", "D1"),
    ("W_CL_07_Marsilea minuta", "Marsilea minuta", "D1"),
    ("W_CL_08_Panicum repens", "Panicum repens", "D1"),
    ("W_CL_09_Paspalum scrobiculatum", "Paspalum scrobiculatum", "D1"),
    ("W_CL_10_Pteris vittata", "Pteris vittata", "D1"),
    ("W_CL_11_Synedrella nodiflora", "Synedrella nodiflora", "D1"),
]

FOLDER_NAMES = [c[0] for c in CLASSES]
SPECIES = [c[1] for c in CLASSES]
NUM_CLASSES = len(CLASSES)
