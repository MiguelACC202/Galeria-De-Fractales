## Fractales de Newton, Raíces complejas
  
  ***Definición*** Cuando la búsqueda de la solución de un problema de aplicación implica la resolución de ecuaciones no lineales se hace uso de métodos numéricos. Siendo el método de Newton uno de los más usados debido a su versatilidad y agilidad, es de gran interés emplearlo especialmente para aproximar soluciones de sistemas de ecuaciones no lineales. Solucionar ecuaciones con variable compleja a través del método de Newton tiene una aplicación muy interesante en el campo de los fractales como es la del problema de Cayley y las figuras fractales que se producen a partir de la convergencia, divergencia e incluso la eficiencia del método. En este artículo se muestra el estudio del problema de Cayley a través de la generalización del método de Newton a $mathds{R}^{2}$. Además, se presentan algunos fractales producidos por iteraciones del método de Newton en los complejos.
 ***Algunos Fractales*** Los siguientes ejemplos de fractales son realizados con codigo Python, donde se relacionan funciones polonomiales y/o trigonometricas.
 **-** los paquetes necesarios para la implementación de los codigos de los fractales son:
 
 *Ejemplo 1* $f(z)= z^{3}+z^{2}$
```
def f(z):
  return z**5+z**2
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-2
xb=2
ya=-2
yb=2
maxit=202
h=1e-6
eps=1e-3
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*8
            g=i*12
            b=i*53
            image.putpixel((x,y),(r,g,b))
image
```
![newton1](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton1.png)

 *Ejemplo 2* $f(z)= z+z^{2}-z^{3}$
 ```
 def f(z):
  return z+z**2-z**3
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-2
xb=2
ya=-2
yb=2
maxit=202
h=1e-6
eps=1e-3
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*46
            g=i*12
            b=i*32
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![newton2](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton2.png)
 
  *Ejemplo 3* $f(z)= \sin(z^{3})$
 ```
 def f(z):
  return np.sin(z**3)
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-3
xb=3
ya=-3
yb=3
maxit=202
h=1e-6
eps=1e-3
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*48
            g=i*12
            b=i*12
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![newton3](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton3.png)
 
 *Ejemplo 4* $f(z)= \cos(z)+z^{5}-\sin(z)$
 ```
 def f(z):
  return np.cos(z)+z**5-np.sin(z)
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-2
xb=2
ya=-2
yb=2
maxit=202
h=1e-6
eps=1e-3
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*63
            g=i*190
            b=i*63
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![newton4](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton4.png)
 
  *Ejemplo 5* $f(z)= (z^{7}-z^{5})*\sin(z)$
 ```
 def f(z):
  return (z**7-z**5)*np.sin(z)
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-2
xb=2
ya=-2
yb=2
maxit=202
h=1e-6
eps=1e-3
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            dz=(f(z+complex(h,h))-f(z))/complex(h,h)
            z0=z-f(z)/dz
            if abs (z0-z)<eps:
                break
            z=z0
            r=i*8
            g=i*8
            b=i*8
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![newton5](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton5.png)
 
 
     
