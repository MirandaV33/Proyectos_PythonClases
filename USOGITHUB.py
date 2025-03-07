# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:57:08 2025

@author: l
"""

##¿Como inicializar y conectar con un repositorio remoto?

echo "# TrabajosPracticos" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/MirandaV33/TrabajosPracticos.git
git push -u origin main

##IMPORTANTE
##Si creo nuevos repositorios, para no tener conflicto TENGO que cerrar el anterior, 
##porque los repositorios remoto se conectan al local 

git remote remove origin

## ¿Como agrego archivos desde mi repositorio local? 

git init 
cd ##DIRECCIONCARPETAENQUEESTAELARCHIVO
git add ##Nombredelarchivo
git add . 
git status 
git commit -m "Agregar archivo al repositorio"
git push -u origin main 


##Actualizar archivos en git
