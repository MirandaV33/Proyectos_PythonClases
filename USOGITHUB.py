# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:57:08 2025

@author: l
"""
##Conectar cuenta de github
git config --global user.name "(usuario)"
git config --global user.email "(mail)"
##Vincular carpeta
cd ##Nombredelacarpeta
git init
git remote add origin ##Linkdelrepositorio !!Recomiendo cerarlo desde la pagina de github mas facil que crearlo desde los comandos
git remote -v
git add . 
git commit "Primer commit"
git push -u origin main 
##Esto sube TODO lo que tengas es tu carpeta 

##¿Como inicializar y conectar con un repositorio remoto?
echo "# Nombre del repositorio remoto" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin ##Linkdelrepositorio
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
git init 
cd ##DIRECCIONCARPETAENQUEESTAELARCHIVO
git status          # Verifica los archivos modificados
git add .           # Agrega todos los archivos al área de preparación
git commit -m "mensaje de cambios"  # Realiza el commit
git push -u origin main   # Sube los cambios al repositorio remoto

