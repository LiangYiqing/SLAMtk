# -*- coding: utf-8 -*- 

import re
import os
import codecs
import random
from sets import Set
#from pylab import *
import subprocess
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
#import numpy as np
#import mdp
from utilsSLAM import *
from resultatsSLAM import *

def anonymisation (nomdufichier, numpat):
    res = charger(nomdufichier)

    montagne = ['Alpes', 'Hautes Alpes', 'Haute Loire', 'Loire']
    nom = ['Eckman', 'Diaz', 'Boileau', 'Maurin']
    capitale = ['Paris']
    pays = ['France', 'Togo', 'Hollande', 'Portugal', 'Qatar']
    ville  = ['Lyon', 'Villeurbane', 'Nancy', 'Nantes', 'Clermont Ferrand', 'Strasbourg', 'Carcassone', 'Meyzieux', 'Toulouse', 'Evreux', 'Clermont']
    departement = ['Savoie']
    institution = ['CRESOP', 'Vinatier']
    prenomF = ['Julia', 'Marion']
    prenomM = ['Jean', 'Fabrice', 'Laurent', 'Boris', 'Martin', 'Pierre', 'Adrien', 'Denis', 'Eric', 'Cameron', 'Paul']

    for indx, vi in enumerate(departement) : # supprimer les departements
        comp = re.compile(vi)
        res = re.sub(comp, 'Departement'+str(indx+1), res)      

    for indx, vi in enumerate(nom) :         # supprimer les noms
        comp = re.compile(vi)
        res = re.sub(comp, 'Nom'+str(indx+1), res)      

    for indx, vi in enumerate(pays) :        # supprimer les pays
        comp = re.compile(vi)
        res = re.sub(comp, 'Pays'+str(indx+1), res)      

    for indx, vi in enumerate(capitale) :    # supprimer les capitales
        comp = re.compile(vi)
        res = re.sub(comp, 'Capitale'+str(indx+1), res)      

    for indx, vi in enumerate(montagne) :    # supprimer les montagnes
        comp = re.compile(vi)
        res = re.sub(comp, 'Montagne'+str(indx+1), res)      

    for indx, vi in enumerate(ville) :       # supprimer les villes
        comp = re.compile(vi)
        res = re.sub(comp, 'Ville'+str(indx+1), res)      

    for indx, vi in enumerate(institution) : # supprimer les institutions
        comp = re.compile(vi)
        res = re.sub(comp, 'Institution'+str(indx+1), res)

    for indx, vi in enumerate(prenomF) :     # supprimer les prenoms feminins
        comp = re.compile(vi)
        res = re.sub(comp, 'PrenomF'+str(indx+1), res)            

    for indx, vi in enumerate(prenomM) :     # supprimer les prenoms masculin
        comp = re.compile(vi)
        res = re.sub(comp, 'PrenomM'+str(indx+1), res)  


    if 'anonym' not in os.listdir('..'):
        os.mkdir('../anonym')
    sauver(res, '../anonym/'+numpat+'anonym.txt')

def normaliseNomdestours(texte, numpat):
    nompatient = re.compile('\*'+numpat+r'\s*?:')
    res = re.sub(nompatient, '\nPa- ', texte)   # supprimer la mention au patient
    res = re.sub(r'\*PSY\s*:','\nPs- ', res)    # supprimer la mention au psy
    return res

def normaliseSanslesnoms(texte, numpat):
    nompatient = re.compile('\*'+numpat+r'\s*?:')
    res = re.sub(nompatient, '\n- ', texte)   # supprimer la mention au patient
    res = re.sub(r'\*PSY\s*:','\n- ', res)    # supprimer la mention au psy
    return res

def normaliseValibel(texte, numpat):
    nompatient = re.compile(r'\*'+numpat+r' :')
    res = re.sub(nompatient, '\nspk1 ', texte)   # supprimer la mention au patient
    res = re.sub(r'\*PSY\s*:','\nspk2 ', res)    # supprimer la mention au psy
    res = re.sub(r'\.+?', ' ', res)
    res = re.sub(r'[\?\,\;\:\!]', ' ', res)
    res = re.sub(r' +', ' ', res)
    return res

#le troisième argument permet soit de produire soit avec la mention Ps ou Pa (1) soit sans (0)
def normalizeheart (nomdufichier, numpat, version=0):
#    nompatient = re.compile(r'\*'+numpat+r' :')
    chaine = charger(nomdufichier)
    res = re.sub(r'@(.)*\n', '', chaine)      # supprimer les meta-donnes
    res = re.sub(r'%(.)*\n', '', res)         # supprimer les commentaires
    res = re.sub(r'\t', '', res)              # supprimer tous les retours a la ligne
    res = re.sub(r'\r\n', ' ', res)           # supprimer tous les retours a la ligne
    res = re.sub(r'\(\s*\d*\s*sec\..*?\)', '...', res) # supprimer les (X sec) 
    res = re.sub(r'\(\s*\d*\s*\.*?\)', '...', res)     # supprimer les (X)
    res = re.sub(r'(\w)([.,;:?!])', '\\1 \\2', res)    # normalisation des espaces devant les ponctuations    
    res = re.sub(r'\w{5,}_\w{5,}', '', res)   # supprimer les marques avec des chiffres
    res = re.sub(r'\[ *?=!*.*?\]', '...', res)# supprimer les commentaires de comportement
    res = re.sub(r'\[/+\]', '/', res)       # remplacer les [/] par / 
    res = re.sub(r'\[[!?.]\]', '...', res)    # remplacer les [!?.] par ... 
    res = re.sub(r'\+/+\.*', '?', res)        # remplacer les /+... par ?
    res = re.sub(r'\+<', '', res)             # supprimer les +<
    res = re.sub(r'\+\.{2,}', '...', res)     # supprimer les +...
    res = re.sub(r'\+\^', '', res)            # supprimer les +^
    res = re.sub(r'\+,', '', res)             # supprimer les +,
    res = re.sub(r'\+\+', '', res)            # supprimer les ++
    res = re.sub(r'\x15', '', res)            # supprimer les caracteres etranges
    res = re.sub(r'[ \t\r\f\v]+',' ', res)    # supprimer les doubles espaces
    res = re.sub(r'\[(.+?)\]', '\\1', res)
    while (re.search(r'<([^>]*?)>', res)!= None):
        res = re.sub(r'<([^>]*?)>', '\\1', res)    # supprimer les <blablabalba> recurssivement
    
    while (re.search(r'\(([^\)]*?)\)', res)!= None):
        res = re.sub(r'\(([^\)]*?)\)', '\\1', res) # supprimer les incertitudes annotateurs recurssivement
    
#    while (re.search(r'\[([^\]]*?)\]', res)!= None):
#        res = re.sub(r'\[([^\]]*?)\]', '\\1', res) # supprimer les incertitudes annotateurs recurssivement

    res = re.sub(r'\n-\s+(\w)', lambda pat: '\n- '+pat.group(1).lower(), res)  # supprimer les espaces en debut de ligne et mise en minuscule en debut de ligne
    res = re.sub(r'\. ([A-Z])([^A-Z])', lambda pat: '. '+pat.group(1).lower()+pat.group(2), res)  # mise en minuscule des debut de phrase
    res = re.sub(r'\s\.\.\.', '...', res)     # supprimer les espaces devant les ...
    res = re.sub(r'\.(\w)', '. \\1', res)     # supprimer les espaces devant les
    res = re.sub(r'\s+([\.\,])', '\\1', res)  # supprimer les espaces devant les .
    res = re.sub(r'\n{2,}', '\n', res)        # supprimer les derniers saut-de-lignes
    res = re.sub(r'\.{4,}','...', res)        # normaliser les ...
    res = re.sub(r'yy+', '', res)             # supprimer les yyyy*
    res = re.sub(r'xx+', '', res)             # supprimer les xx*
    res = re.sub(r'euh,', 'euh', res)         # supprimer les Euh,
    #res = re.sub(r"' ", "'", res)
    if version == 1 :
        res = normaliseNomdestours(res, numpat)
    elif version == 2 :
        res = normaliseValibel(res, numpat)
    else :
        res = normaliseSanslesnoms(res, numpat)
    res = res[2:]
    return res

def normalize (nomdufichier, numpat, version=0):
    res = normalizeheart(nomdufichier, numpat, 0)
    res2 = normalizeheart(nomdufichier, numpat, 1)
    res3 = normalizeheart(nomdufichier, numpat, 2)
    if 'normalise' not in os.listdir('..'):
        os.mkdir('../normalise')
    if 'TourParole' not in os.listdir('../normalise'):
        os.mkdir('../normalise/TourParole')
    if 'Anon' not in os.listdir('../normalise'):
        os.mkdir('../normalise/Anon')
    if 'Valibel' not in os.listdir('../normalise'):
        os.mkdir('../normalise/Valibel')

    sauver(res2, '../normalise/TourParole/'+numpat+'out.txt')
    sauver(res, '../normalise/Anon/'+numpat+'out.txt')
    sauver('<deb id="'+os.path.split(os.getcwd())[0]+'/normalise/Valibel/'+numpat+'out.txt">\n'+res3.lower()+'\n<fin id="'+os.path.split(os.getcwd())[0]+'/normalise/Valibel/'+numpat+'out.txt">', '../normalise/Valibel/'+numpat+'out.txt')
    return res


# Afficher tous les mots qui ne sont pas en debut de phrase avec une majuscule
def affineranonymisation (res):
    return re.findall(r'[A-Z]\w\w*', res)


# annonymise et normalise
def anonETnormalisation (nomdufichier, numpat, version=0):
    print 'hhehhehehheheheheh'
    anonymisation(nomdufichier, numpat)
    return normalize('../anonym/'+numpat+'anonym.txt', numpat, version)


# production d'une ressource fusionnant tous les textes aléatoirement (pour l'analyse syntaxique ET mémorisation du random dans un fichier memoire
def pref(x,y) : 
    return x+y

def pref2(x,y) : 
    return x+str(y)

#creation de la table memorisant le numéro du fichier - tour de parole de la ressource randomisée à partir des documents deja produits

def memorisation(res, nomdurep):
    res2 = res
    sauver(''.join(res), nomdurep+'RAI1Tp.txt')
    sauver(''.join(map(lambda x:re.sub(r'\d+\w*\d+(.*)\n', '\\1\n', x), res2)), nomdurep+'RAI1.txt')
    sauver(''.join(map(lambda x:re.sub(r'(\d+\w*\d+)P.-(.*)\n', '\\1\\2\n', x), res2)), nomdurep+'RAI1Anon.txt')
    sauver(''.join(map(lambda x:re.sub(r'\d+\w*\d+P.-(.*)\n', '\\1\n', x), res2)), nomdurep+'RAI1TpAnon.txt')

def memorisationtable(fichier, nomdufichiertable, vers=0):
    if vers:
        res = map(lambda x:re.sub(r'(\d+\w*\d+).*-.*\n', '\\1', x), fichier)
    else :
        res = map(lambda x:re.sub(r'(\d+\w*\d+).*\n', '\\1', x), fichier)
    sauver('\n'.join(res)+'\n', nomdufichiertable)

def memorisationtable2(fichier, nomdufichiertable, vers=0):
    if vers:
        res = map(lambda x:re.sub(r'(\d+\w*\d+.*?)-.*\n', '\\1', x), fichier)
    else :
        res = map(lambda x:re.sub(r'(\d+\w*\d+.*?)-.*\n', '\\1', x), fichier)
    sauver('\n'.join(res)+'\n', nomdufichiertable)

def toutenvrac (nomrepertoire, version=0):
    import random
    res = []
    for nomdufichier in listDirectory(nomrepertoire+'TourParole', '.txt'):
        with codecs.open(nomdufichier, encoding="UTF-8") as F:
            contenu = F.readlines()
            #print len(contenu)
            contenu[len(contenu)-1]=contenu[len(contenu)-1]+'\n'
            nom = [re.search(r'.*/(.*?)\..*', nomdufichier).group(1)]*len(contenu)
            nomindice = map(pref2, nom, map(lambda x : str(x+1).rjust(4,'0'), range(len(contenu))))

            res = res + map(pref, nomindice, contenu)

    random.shuffle(res)
    memorisationtable(res, 'AllInOne/tablerandom.txt', vers = version)
    memorisationtable2(res, 'AllInOne/tablerandom2.txt', vers = version)
    memorisation(res, 'AllInOne/')

# Il faut produire : allInOne, randomAllInOne, tablerandomAllInOne
#toutenvrac('../normalise/')

# aller cherche tous les fichier *.cha, les anonymiser (produit les *anonym.txt. dans le repertoire anonym), les normaliser (produit *.txt dans le repertoire normalise). Attention, il faut anonymiser avant de normaliser

def annoEtnormalisationAll(rep, ext, version=0):
    for fichier in listDirectoryrec(rep, [ext], []):
        print fichier

        #print re.search(r'\w*/(\w*)'+ext, fichier).group(1)
        #if os.path.isfile(fichier):
        print os.path.split(fichier)[1].split('.')[0]
        anonETnormalisation(fichier, os.path.split(fichier)[1].split('.')[0], version)


# afficher les mots avec majuscule, ce qui améliore l'anonymisation 

def afficherPBAnonym(rep):
    aanon = Set([])
    for fichier in listDirectoryrec(rep, ['.txt'], []):
        aanon.update(Set(affineranonymisation(charger(fichier))))
    print aanon

#----------------------------------------------------------
# production des fichiers individuels anonym et normalisés
#annoEtnormalisationAll('../corpus/', '.cha')

#production des memes fichiers avec mention de qui prend le tour de parole
#annoEtnormalisationAll('../corpus/', '.cha', 1)

# afficher les problemes sur l'anonymisation
#afficherPBAnonym('../anonym/')
#afficherPBAnonym('../normalise/') #plus lisible a partir de la normalisation

# production de fichiers qui contiennent tout, randomise par tour de parole,avec plus ou moins d'information explicitent,  ainsi que la table de mémorisation de l'ordre des tour de parole et le fichier associant explicitement les deux

# production du fichier brut sans les disfluences
def touslesfichiersPOURdisfluences(directory, fichier):
    res = charger('AllInOne/RAI1.txt')
    res = re.sub('Pa- ', 'spk1 ', res)      # supprimer la mention au patient
    res = re.sub(r'Ps- ','spk2 ', res)      # supprimer la mention au psy
    res = re.sub(r'\.+?', ' ', res)         # supprimer les . .. ... ....
    res = re.sub(r'[\?\,\;\:\!]', ' ', res) # supprimer la ponctuation
    res = re.sub(r' +', ' ', res)           # supprimer les espaces
    # res = re.sub(r'\{.*?\}', '', res)
    # print re.sub(r'.*\/(.*?)\..*', '\\1', fichier)
    sauver('<deb id="/Users/Home/Downloads/M1/AllInOne/RAI1PrePourDisfluences.txt">\n'+res+'<fin id="/Users/amblard/Desktop/redaction/SLAM/ressource/AllInOne/RAI1PrePourDisfluences.txt">', fichier)

def nettoyagedisfluences(fichier):
#    for fichier in listDirectoryrec(directory, ['.snt'], []):
    res = charger(fichier)
    res = re.sub(r'\{.*?\}', '', res)
    res = re.sub(r' {2,}', ' ', res)
    res = re.sub(r"' ", "'", res)
    res = re.sub(r'\n +?(.)', lambda pat: '\n'+pat.group(1).lower(), res)  # supprimer les espaces en debut de ligne et mise en minuscule en debut de ligne
    res = res[2:]
    sauver(res,'AllInOne/'+re.sub(r'.*\/(.*?)\..*', '\\1', fichier)+'SansDisfluences.txt')
    res = chargerlignes('AllInOne/'+re.sub(r'.*\/(.*?)\..*', '\\1', fichier)+'SansDisfluences.txt')
    #res = '\n'.split(res)
    #print res
    res.sort()
#    print res
    res = ''.join(res)
    sauver(res, 'AllInOne/sort.txt')


def deuxstring(l1, l2):
    return str(re.sub(r'(.*)\n', '\\1', l1))+str(re.sub(r'(.*)\n', '\\1', l2))

def deuxstrings(l1, l2):
    res = ''
    for i in range(len(l1)):
        res = res+re.sub(r'(.*)\n', '\\1', l1[i])+l2[i]
    return res

def reconstructionDisfluences(fichier):
    res = chargerlignes('AllInOne/RAI1PrePourDisfluences_norm.snt')
    res = res[1:-1]
    table = chargerlignes('AllInOne/tablerandom2.txt')
    #print 'nombre de tour de parole: '+str(len(res))
    #print 'taille de la table : '+str(len(table))
    sauver(deuxstrings(table, res), fichier)

def sortDisfluences():
    return 'a'

#produire les analyses des disfluences grace a Distagger. Arguments : le chemin vers le repertoire qui continet distagger, le fichier sur lequel on souhaite avoir les annotations en disfluence, le répertoire dans lequel on stocke les resultats.
def Distagger(cheminDistagger, fichier, dest):
    os.system('java -jar '+cheminDistagger+'build/jar/Distagger-0.2.jar -e '+cheminDistagger+'data/euh.txt -o . -i  '+cheminDistagger+'data/insertions.txt -c  '+cheminDistagger+'data/autocorrections.txt -m '+cheminDistagger+'data/meta.txt -t '+fichier)
    os.system('mv *.snt '+dest)
    os.system('mv *.rha '+dest)

def somme (liste):
    res = 0
    for i in liste :
        res = res+(i-3)
    return res





def verification (fichier, resfin2or):
    res = chargerlignes(fichier)
#    som = 0
#    som = som + len(map(lambda x : re.findall(r'IGN+REP', x), res))
#    som = som + len(map(lambda x : re.findall(r'IGN+EUH', x), res))
#    som = som + len(map(lambda x : re.findall(r'IGN+CORR', x), res))
#    som = som + len(map(lambda x : re.findall(r'IGN+FRAG', x), res))
#    print som

    maxi = []
    for i in res :
        maxi.append(int(i[:3]))

    resfin = []
    for i in range(1,max(maxi)+1):
        exp = re.compile('^'+str(i).rjust(3, '0'))
        resfin.append(filter(exp.search, res))

    expPs = re.compile('s{')
    expPa = re.compile('a{')

    resfin2 = []
    for text in resfin :
        resfin2.append((filter(expPs.search, text), filter(expPa.search, text)))
    print "est-ce que les données sont les mêmes ?"
    print resfin2 == resfin2or

    nbtagS = 0
    nbmot = []
    nbmot2 = []
    for i in resfin2 :
        nbtagS = nbtagS+plus(map(lambda x : len(re.findall(r'IGN\+REP', x)), i[1]))+plus(map(lambda x : len(re.findall(r'IGN\+EUH', x)), i[1]))+plus(map(lambda x : len(re.findall(r'IGN\+CORR', x)), i[1]))+plus(map(lambda x : len(re.findall(r'IGN\+FRAG', x)), i[1]))
        
        nbmot = nbmot+[sum(map(lambda x : len(x.split()), i[1]))]
        nbmot2 = nbmot2 + [somme(map(lambda x : len(x.split()), i[1]))]

    nbtagS12 = plus(map(lambda x : len(re.findall(r'IGN\+REP', x)), resfin2[13][1]))+plus(map(lambda x : len(re.findall(r'IGN\+EUH', x)), resfin2[13][1]))+plus(map(lambda x : len(re.findall(r'IGN\+CORR', x)), resfin2[13][1]))+plus(map(lambda x : len(re.findall(r'IGN\+FRAG', x)), resfin2[13][1]))

    print 'nombre de tag dans l entretien du patient 13 (premier temoin) '
    print nbtagS12
    print 'nombre de mots dans l entretien  du patient 13'
    print nbmot2[12]


    print "nombre de tags en tout dans la vérification"
    print nbtagS
    print "nombre de mots tot par entretien dans la vérification"
    print nbmot2
    print "nombre de mots tot dans la vérification"
 #   print sum(nbmot)
    print sum(nbmot2)
    print "ratio côté vérification"
    print float(nbtagS)/plus(nbmot2)

def plus (l):
    res = 0
    for i in l :
        res = res+i
    return res

#verification('AllInOne/fusion.txt')



    # affichage est un dictionnaire dont la clé est le numero du patient. Pour chacun il y a 3 vecteurs un pour les positions des EUH, le second pour les positions des REP, le troisieme pour les positions des EUH+REP:

# calcul d'une analyse en composantes principales

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    mn = NP.mean(data, axis=0)
    # mean center the data
    data -= mn
    # calculate the covariance matrix
    C = NP.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = LA.eig(C)
    # sorted them by eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:,:dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    data_rescaled = NP.dot(evecs.T, data.T).T
    # reconstruct original data array
    data_original_regen = NP.dot(evecs, dim1).T + mn
    return data_rescaled, data_original_regen



def calculacp(mat2, tag, nompat, S2, T2, matrice):

    mat2Ps = []
    mat2Pa = []
    mat2type = []
    for i, val in enumerate(mat2):
        if i == 0 :
            mat2type = val
        if i%2 == 0 and i!=0 :
            mat2Ps.append(val)
        if i%2 == 1 :
            mat2Pa.append(val)
    
    mat = []
    for i in matrice :
        mat.append(i[2:])

    print transpose(mat2)

    data = np.array(transpose(mat2[1:]))
    #data = np.array(transpose(mat2Ps))
    #data = np.array(transpose(mat2Pa))
    #data = np.array(mat)
        
    C = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eig(C)


    print '------'
    print eigenvalues

    # 2D PCA - get the two eigenvectors with the largest eigenvalues
    v1, v2 = eigenvectors[:,:2].T
    print v1
    print v2

    # Project the data onto the two principal components
    data_pc1 = np.dot(data, v1)
    data_pc2 = [np.dot(v2, d) for d in data]

    # Scatter plot in PCA space
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data_pc1, data_pc2, 'x')
    ax.set_xlabel(r'$PC_1$')
    ax.set_ylabel(r'$PC_2$')
    ax.legend(['data'])
    plt.show()


def plot_pca(data):
    clr1 =  '#2026B2'
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:,0], data_resc[:,1], '.', mfc=clr1, mec=clr1)
    plt.show()

def calculacp3 (mat2, tag):

    #Create sample data
    var1 = np.random.normal(loc=0., scale=0.5, size=(10,5))
    var2 = np.random.normal(loc=4., scale=1., size=(10,5))
    var = np.concatenate((var1,var2), axis=0)

    #Create the PCA node and train it
    pcan = mdp.nodes.PCANode(output_dim=3)
    pcar = pcan.execute(np.array(mat2[1:]))

    #Graph the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pcar[:10,0], pcar[:10,1], 'bo')
    ax.plot(pcar[10:,0], pcar[10:,1], 'ro')

    #Show variance accounted for
    ax.set_xlabel('PC1 (%.3f%%)' % (pcan.d[0]))
    ax.set_ylabel('PC2 (%.3f%%)' % (pcan.d[1]))

    plt.show()

#    res1 = []
 #   for i, val in enumerate(tag) :
#        if i%2 == 0:
#            res1.append(val+'Ps')
#        else :
#            res1.append(val+'Pa')
#
#    res = []
#    for i, val in enumerate(mat2[1:]):
#        res.append(['Pat'+str(i)]+val)
#    
#    points = np.array(res1+res)
#
#    myPCA = PCA(dim+points)

    #myPCA = PCA(dataMatrix)   
    #myPCA.numrows == len(dataMatrix)   #True. The number of rows in the data matrix 
    #myPCA.numcols == len(dataMatrix[i])  #True for all valid index i.  The number of columns in any row of the data matrix 

    #pcDataPoint = myPCA.project(aDataPoint)   # pcDataPoint is the same point as aDataPoint, but in terms of the PC axes.
    
    #myPCA.center(x) == (x -myPCA.mu)/myPCA.sigma  # True, note that subtraction and division are on an element by element basis
    #myPCA.center(myPCA.mu+myPCA.sigma) == [1, 1, 1, ...]   #True. one standard deviation away in all measurement directions.




def calculacp2 (mat2):

    all= np.array(mat2[1:])

#    pcan = mdp.nodes.PCANode(output_dim=3)
#    pcar = pcan.execute(mat)

#    all = numpy.array (fr + en)
    node = mdp.nodes.PCANode()
    node.train (all)
    y    = node (all)
    # construction du nuage de points
    # ACP
    # construction de l’ACP
    # on peut aussi écrire y = mdp.pca (all)
    # obtention des coordonnées des points dans le plan de projection
    frcx = [ y [i,0] for i in range (0, nbfr) ]
    frcy = [ y [i,1] for i in range (0, nbfr) ]
    encx = [ y [i,0] for i in range (nbfr, nbfr + nben) ]
    ency = [ y [i,1] for i in range (nbfr, nbfr + nben) ]


    plt.plot (frcx, frcy, "rx",ms=10,mew=2.5) # on trace la courbe des textes français
    plt.plot (encx, ency, "bv")        # on trace la courbe des textes anglais
#    pylab.legend (("francais", "anglais"), loc=2)  # légende (sans accent)
#    pylab.title ("langue")
    plt.xlabel ("frequence de W")
    plt.ylabel ("frequence de H")
    #plt.savefig ("graphe.png")
    plt.show ()
#    #Graph the results
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(pcar[:10,0], pcar[:10,1], 'bo')
#    ax.plot(pcar[10:,0], pcar[10:,1], 'ro')
#    
    #Show variance accounted for
 #   ax.set_xlabel('PC1 (%.3f%%)' % (pcan.d[0]))
 #   ax.set_ylabel('PC2 (%.3f%%)' % (pcan.d[1]))

    plt.show()


#    sauver(''.join(resfin2[0][0]), 'AllInOne/test.txt'

         
#    taille = 0
#    for i in resfin :
#        taille += len(i)
#    print taille

#    print resfin[3]
    #for i in res :
    #    if re.match(exp,i )!= None :
            #print i
    #sauver(''.join(res), 'AllInOne/sortavecdisfluences.txt')

#listepsy = ['001PEP','002LOA','003DOC','004KAT','005AUB','006MAB','007THC','008BAL','009MAE','010BLD','011HAM','012PEJ','035RIP','036DEF','037VOA','038BAN']
#listetemoins = ['013DUR','014BRL','015HOS','016FAR','028MAH']



#----------------------------------------------------------
# production des fichiers individuels anonym et normalisés
#annoEtnormalisationAll('../corpus/', '.cha')

# #production des memes fichiers avec mention de qui prend le tour de parole
annoEtnormalisationAll('../corpus/', '.cha', 1)
#
# # afficher les problemes sur l'anonymisation
# #afficherPBAnonym('../anonym/')
#
# #plus lisible a partir de la normalisation
# #afficherPBAnonym('../normalise/')
# # Pour améliorer l'anonymisation il faut ajouter les termes a retirer directement dans le code de la fonction anonymisation
#
# # extrait tout le contenu du repertoire 'TourdeParole' (du repertoire passé en parametre) et produit des fichiers ou les tours de parole sont randomisés (dans le repertoire AllInOne
toutenvrac('../normalise/')
#
# # sauve dans le second parametre le fichier nettoyé pour passer Distagger
# touslesfichiersPOURdisfluences('../normalise/Valibel', 'AllInOne/RAI1PrePourDisfluences.txt')
#
# #Applique Distagger sur le ficher fait pour.
# #Distagger('distagger-0.2/','AllInOne/RAI1PrePourDisfluences.txt', 'AllInOne/')
#
# # concatene les tours de paroles avec les identifiants dans fusion.txt
# reconstructionDisfluences('AllInOne/fusion.txt')
#
# #reconetruit les entretiens
# nettoyagedisfluences('AllInOne/fusion.txt')
#
# prefiltre('AllInOne/fusion.txt')
#
# # extraction des informations de Distagger : construction des graphiques (figures), des documens pdf (resultats)
# gestiondisfluences('AllInOne/fusion.txt','.', '..', 0)

def test(fichier):
    res = chargerlignes(fichier)
    ress = ''
    for i in res :
        if re.search(r'\d{4}Ps\{', i):
           ress = ress+i 
    sauver(ress, 'AllInOne/test.txt')

#test('AllInOne/fusion.txt')

def alldistag(res, rep):

    res = re.sub('Pa- ', 'spk1 ', res)      # supprimer la mention au patient
    res = re.sub(r'Ps- ','spk2 ', res)      # supprimer la mention au psy
    res = re.sub(r'\.+?', ' ', res)         # supprimer les . .. ... ....
    res = re.sub(r'[\?\,\;\:\!]', ' ', res) # supprimer la ponctuation
    res = re.sub(r' +', ' ', res)           # supprimer les espaces
    sauver('<deb id="/Users/Home/Downloads/M1/AllInOne/RAI1PrePourDisfluences.txt">\n'+res+'<fin id="/Users/amblard/Desktop/redaction/SLAM/ressource/AllInOne/RAI1PrePourDisfluences.txt">', fichier)

    if 'resdistag' not in os.listdir('.'):
        os.mkdir('resdistag')
    for f in rep :
        Distagger('distagger-0.2/',f, 'resdistag')

