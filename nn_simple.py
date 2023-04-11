#
# Exemple d'un réseau de neurones très simple
# adapté de https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#
# Cet exemple emprunte certaines bonnes pratiques, mais est loin d'être parfait
# Il vous donnera la base nécessaire pour implémenter l'entrainement par algorithme évolutif
#

#%%
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#%% # MNIST bug, on doit aller le chercher manuellement (python - HTTP Error 503: Service Unavailable whan trying to download MNIST data. (s. d.). Stack Overflow. https://stackoverflow.com/questions/66646604/http-error-503-service-unavailable-whan-trying-to-download-mnist-data)
# on a seulement besoin de l'exécuter une fois
# on va faire des appels natifs
if not os.path.exists("MNIST"):
    if "linux" in sys.platform:
        os.system("wget www.di.ens.fr/~lelarge/MNIST.tar.gz")
        os.system("tar -zxvf MNIST.tar.gz")
        os.system("rm MNIST.tar.gz")
    elif "win32" in sys.platform: 
        os.system("pwsh -command $ProgressPreference = 'SilentlyContinue'")
        os.system('pwsh -command "Invoke-WebRequest http://www.di.ens.fr/~lelarge/MNIST.tar.gz -OutFile MNIST.tar.gz"')
        os.system('pwsh -command "tar -zxvf MNIST.tar.gz"')
        os.system('pwsh -command "rm MNIST.tar.gz"')
    else:
        print("tough luck buddy!")

# %%
# Chargement du dataset MNIST
# on va appliquer les opérations de conversion ici pour fins de rapidité
train_data = datasets.MNIST(
    root="./", # le dossier racine où se trouve le dossier MNIST 
    train=True, # les images d'entrainement
    download=False, # pas besoin de télécharger
    transform=Compose([
        ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        Lambda(lambda x : torch.flatten(x))]) # on "écrase" l'image pour retourner un vecteur contenant les pixels
    )

test_data = datasets.MNIST(
    root="./", 
    train=False, # les images de validation
    download=False,
    transform=Compose([
        ToTensor(), # https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor
        Lambda(lambda x : torch.flatten(x))]) # https://pytorch.org/docs/stable/generated/torch.flatten.html
    )


# %%
# pour entrainement éventuel sur gpu..., vu la taille du réseau, ça ne vaut probablement pas la peine...
device = "cuda" if torch.cuda.is_available() else "cpu"

# Comme on ne va pas entrainer en utilisant le SGD (ou autre) on a pas besoin de calculer automatiquement
# le gradient durant les forward pass -- en gros, on sauve du temps de calcul
# si vous voulez comparer à un entrainement classique vous devrez le remettre à True pour votre boucle d'entrainement classique
torch.autograd.set_grad_enabled(False)

class NeuralNet(nn.Module):
    """Implémente un réseau de neurones linéaire très simple (perceptron multicouche),
       inspiré de celui de 3blue1brown.

       trois couches pleinement connectées
       
       784 -> 16 -> 16 -> 10

       activation sigmoide et logits obtenus à l'aide de LogSoftMax

       les gradients sont désactivés pour permettre à l'étudiant d'entrainer
       le réseau à l'aide de métaheuristiques.

    Args:
        nn (torch.nn.Module): hérite de cette classe, (pas obligatoire, mais ça facilite les choses)
    """
    def __init__(self):
        """Initialise le réseau de neurones
        """
        super().__init__()
        
        # les différentes couches
        self.fc1 = nn.Linear(28*28, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

        # la fonction pour les logits (retourne le "score" des classes)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        # la fonction pour calculer le loss (l'erreur de prédiction)
        self.loss_fn = nn.CrossEntropyLoss()
        self.float() # Pas besoin de double precision, beaucoup plus rapide aussi...

    #@torch.no_grad()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Calcule le résultat du réseau de neurones sur une ou plusieurs images.

        Args:
            x (torch.Tensor): l'image d'entrée, normalisée et écrasée, taille [B,N] où:
            - B est le batch size
            - N le nombre de pixel

        Returns:
            torch.Tensor: le résultat de taille [B,C] du traitement par le réseau où:
            - B est le batch size
            - C est le nombre de classes 
        """
        # on "applatit" l'image, le -2 indique que l'on joins les deux dernières dimensions, 
        # la largeur et la longueur pour que les pixels soient dans la même dimension 
        x = torch.relu(self.fc1(x)) # activation sur la première couche
        x = torch.relu(self.fc2(x)) # activation sur la deuxième couche
        x = self.fc3(x) # calcul de la dernière couche
        return x

    def loss(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """Fonction pour retourner le loss de notre réseau sur un ensemble de prédictions.

        Args:
            x (torch.Tensor): le tenseur avec les prédictions
            y (torch.Tensor): la solution

        Returns:
            torch.Tensor: la perte calculée selon le cross entropy loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        """
        return self.loss_fn(x, y)

    def get_weights_and_bias(self) -> torch.Tensor:
        """Fonction utilitaire qui retourne un vecteur concaténant les poids et biais de notre réseau de neurones.

        Returns:
            torch.Tensor: un vecteur de dimension 16*784+16*16+10*16+16+16+10 qui représente une concaténation aplatie
             des poids des couches fc1,fc2,fc3 et des biais des couches fc1,fc2,fc3
        """
        return  torch.cat((
            self.fc1.weight.data.flatten(),
            self.fc2.weight.data.flatten(), 
            self.fc3.weight.data.flatten(), 
            self.fc1.bias.data.flatten(), 
            self.fc2.bias.data.flatten(), 
            self.fc3.bias.data.flatten()
            ))

    #@torch.no_grad()
    def set_weights_and_bias(self, x:torch.Tensor) -> None:
        """Fonction utilitaire pour mettre à jour les poids et les biais du réseau de neurones
           à partir d'un individu issu d'un algorithme d'optimisation quelconque.

           On va donc extraire et remettre en forme les sections respectives du vecteur et les assigner aux couches correspondantes

        Args:
            x (torch.Tensor): un vecteur de dimension 16*784+16*16+10*16+16+16+10 qui représente une concaténation aplatie
             des poids des couches fc1,fc2,fc3 et des biais des couches fc1,fc2,fc3
        """
        # les index des fin poids et des biais
        iw1,iw2,iw3,ib1,ib2,ib3 = 12_544, 12_800, 12_960, 12_976, 12_992, 13_002
        self.fc1.weight.data = x[0:iw1].reshape(16, 784)
        self.fc2.weight.data = x[iw1:iw2].reshape(16, 16)
        self.fc3.weight.data = x[iw2:iw3].reshape(10, 16)
        self.fc1.bias.data = x[iw3:ib1]
        self.fc2.bias.data = x[ib1:ib2]
        self.fc3.bias.data = x[ib2:ib3]

    #@torch.no_grad()
    def fast_fonction_objective(self, x:torch.tensor, targets:torch.tensor) -> torch.Tensor:
        """utiliser cette fonction pour le calcul du score du réseau de neurones

        Args:
            x (torch.tensor): un tenseur de taille BxN contenant les données pour évaluation/entrainement
            targets (torch.tensor): un vecteur de taille B contenant la classe de chaque image

        Returns:
            torch.Tensor: le loss/le score du réseau de neurones
        """
        # on calcule l'erreur de prédiction
        return self.loss(self(x), targets) # le loss est le résultat de la fonction objective, on cherche à le minimiser

#%%
# Fonction d'évaluation de la performance de notre réseau de neurones
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def test(dataloader, model): 
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += model.loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Erreur de test: \n Exactitude: {(100*correct):>0.1f}%, Loss moyen: {test_loss:>8f} \n")        


# %%
### Le code à partir de ce point pourrait probablement se retrouver dans un autre fichier pour rendre la lecture/ gestion plus facile
mnistNN = NeuralNet()


#%%
# Création des dataloader
# On va utiliser les dataloader pour charger les images dynamiquement et appliquer les transformations désirées.
# Dans notre cas, la transformation est torchvision.transforms.ToTensor()
# C'est la façon privilégiée de faire, en particulier lorsqu'on a de grosses bases de données qui ne peuvent pas être complètement stockées en mémoire vive.
# On va tricher sur le batch size pour simplifier l'entrainement : on va tout charger en mémoire (ce sont des petites images donc ça va aller)
train_size = 60_000
test_size = 10_000
train_dataloader = DataLoader(train_data, batch_size=train_size, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=test_size, pin_memory=True)

MINI_BATCH_SZ = 64

# %%
# Comme le dataloader est lent et qu'on peut stocker MNIST en mémoire, on va le faire pour 
# rendre le calcul de la fonction objective beaucoup plus rapide.
# Il est donc recommandé d'utiliser la fonction fast_fonction_objective pour obtenir le loss 
train_set = (train_dataloader.dataset.data.flatten(-2)/255.)
train_targets = train_dataloader.dataset.targets

# pour calculer le loss/fitness du réseau sur les images
# appelez 
mnistNN.fast_fonction_objective(train_set, train_targets)
# remarquez que si vous voulez évaluer plusieurs individus en même temps que vous pourriez instancier plusieurs NeuralNet...
# ce n'est pas vraiment possible pour les réseaux énormes, mais dans ce cas ça devrait aller...

#%%
##### IMPLÉMENTEZ VOTRE ALGORITHME ET ROUTINE D'ENTRAINEMENT ICI 
# en gros je vous suggère d'utiliser la logique de mini-batch
# ça devrait ressembler à quelquechose du genre
# NOTE: ça va bugger comme il y un paquet de variables non déclarées et que c'est pour fin de démonstration seulement

# on va court circuiter un peu le concept d'époque d'entrainement et de boucle de minibatch
num_mini_batch = int(train_size / MINI_BATCH_SZ) # si ce ne sont pas des multiples, on "drop" le reste de la division, e.g. on ne veut pas mettre à jour les poids basé sur une seule image...
idcs = np.arange(train_size)

for i in range(max_generation):

    # à chaque époque on veut brasser l'ordre des images pour garantir une meilleure généralisation
    np.random.shuffle(idcs)
    
    for j in range(num_mini_batch):
        # logique population si approprié
        # ... 

        # boucle d'évaluation de la qualité des individus
        for k in range(taille_population):

            mnistNN.set_weights_and_bias(individu[k])
            qualité_individu[k] = mnistNN.fast_fonction_objective(train_set[idcs[j*MINI_BATCH_SZ:(j+1)*MINI_BATCH_SZ]], train_targets[idcs[j*MINI_BATCH_SZ:(j+1)*MINI_BATCH_SZ]])


        # logique algorithme évolutif 
        # ...
        # ...
        # ...

    # on a fait une époque, on peut peut être afficher des stats ou qqchose du genre, ou pas...
    # on devrait idéalement au moins tester le training et le validation loss (on peut tricher ici et le faire sur le validation loss sur le test_set...)









