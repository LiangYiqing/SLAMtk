# -*- coding: utf-8 -*- 

#import re
import os
import codecs
#import random
from sets import Set
import re



def charger(nomdufichier):
#    filin  = open(nomdufichier, 'r')
    with codecs.open(nomdufichier, encoding="UTF-8") as F:
        return F.read()

def chargerlignes(nomdufichier):
    with codecs.open(nomdufichier, encoding="UTF-8") as F:
        return F.readlines()
#def openfile(filename):
#    with codecs.open(filename, encoding="UTF-8") as F:
#        contents = F.read()

def sauver(texte, nomdufichier):
    #os.makedirs(nomdufichier)
    filout = open(nomdufichier, 'w')
    texte = texte.encode('utf-8')
    filout.write(texte)
    filout.close()


# liste des fichiers ayant l'extension passée en argument, récursivement, en enlevant le fichier spécifique mac

def listDirectory(directory, fileExtList):                                        
    "get list of file info objects for files of particular extensions" 
    fileList = [os.path.normcase(f)
                for f in os.listdir(directory)]             
    Filelist2 = [os.path.join(directory, f)
               for f in fileList
                if os.path.splitext(f)[1] in fileExtList] 
    res = []
    for i in Filelist2:
        for ext in fileExtList :
            if ext in i :
                res.append(i)
    return res
   
# récursion pour la fonction listDirectory
def listDirectoryrec(directory, fileExtList, res):
    res +=listDirectory(directory, fileExtList)
    for i in  os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, i)) :
            listDirectoryrec(os.path.join(directory, i), fileExtList, res)
    return res


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

def plus (l):
    res = 0
    for i in l :
        res = res+i
    return res
