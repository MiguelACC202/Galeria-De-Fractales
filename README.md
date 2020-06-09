<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Fractales de Newton, Raíces complejas

## ***Definición*** 
Cuando la búsqueda de la solución de un problema de aplicación implica la resolución de ecuaciones no lineales se hace uso de métodos numéricos. Siendo el método de Newton uno de los más usados debido a su versatilidad y agilidad, es de gran interés emplearlo especialmente para aproximar soluciones de sistemas de ecuaciones no lineales. Solucionar ecuaciones con variable compleja a través del método de Newton tiene una aplicación muy interesante en el campo de los fractales como es la del problema de Cayley y las figuras fractales que se producen a partir de la convergencia, divergencia e incluso la eficiencia del método. En este artículo se muestra el estudio del problema de Cayley a través de la generalización del método de Newton a $\mathbb R^{2}$. Además, se presentan algunos fractales producidos por iteraciones del método de Newton en los complejos.
  
### ***Algunos Fractales*** 
 Los siguientes ejemplos de fractales son realizados con codigo Python, donde se relacionan funciones polonomiales y/o trigonometricas.
 
**-** los paquetes necesarios para la implementación de los codigos de los fractales son:
 ```
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
 ```
 
#### *Ejemplo 1* - $f(z)= z^{3}+z^{2}$
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

#### *Ejemplo 2* - $f(z)= z+z^{2}-z^{3}$
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
 
#### *Ejemplo 3* - $f(z)= \sin(z^{3})$
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
 
#### *Ejemplo 4* - $f(z)= \cos(z)+z^{5}-\sin(z)$
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
 
#### *Ejemplo 5* - $f(z)= (z^{7}-z^{5})*\sin(z)$
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
 

# Conjuntos De Julia
 
## ***Definición*** 
Los conjuntos de Julia, así llamados por el matemático Gaston Julia, son una familia de conjuntos fractales que se obtienen al estudiar el comportamiento de los números complejos al ser iterados por una función.

Una familia muy importante de conjuntos de Julia se obtiene a partir de funciones cuadráticas simples,como por ejemplo: $Fc(z) = z2 + c$  , donde $c$  es un número complejo. Como por ejemplo:
$$f(z)=z^{2}+c, donde c=-0.8,+0.156i$$
![juliaejemplo](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/ejemplojulia.png)

El conjunto de Julia que se obtiene a partir de esta función se denota **$Jc$**. El proceso para  obtener este conjunto de Julia de  es el siguiente:
Se elige un número complejo cualquiera z y se va construyendo una sucesión de números de la siguiente manera:

$$z_{0} = z$$
$$z_{1} = F(z_{0})= z_{02} + c$$
$$z_{2} = F(z_{1}) =z_{12}+c$$
$$z_{n+1} =  F(z_{n}) =z_{n2}+c$$

   Si esta sucesión queda acotada, entonces se dice que z pertenece al conjunto de Julia de parámetro $c$, denotado por **$Jc$**; de lo contrario, si la sucesión tiende al infinito, z queda excluido de éste. Es fácil deducir que obtener un conjunto de Julia resulta muy laborioso, pues el proceso anterior habría que repetirlo para cualquier número complejo z, e ir decidiendo en cada caso si dicho número pertenece o no al conjunto **$Jc$**. Debido a la infinidad de cálculos que se necesitaban  para  obtener la gráfica correspondiente, se tuvo que esperar hasta los años ochenta para poder  representar estos conjuntos. Pero gracias a todos los avances computacionales se logro porfin verlos en una pantalla, lastimosamente Gaston Julia no alcanzo a verlo por si mismo:
### ***Algunos Fractales*** 
 Los siguientes ejemplos de fractales son realizados con codigo Python, donde se relacionan funciones polonomiales y/o trigonometricas.

**-** los paquetes necesarios para la implementación de los codigos de los fractales son:
 ```
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
 ```
 
#### *Ejemplo 1* - $f(z)= z^{5}+c, donde \^[c=-1-i\]$
```
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-1.5
xb=1.5
ya=-1.5
yb=1.5
maxit=30
def f(z):
    return z**5+complex(-1,-1)
for y in range (imgy):
    zy=y*(yb-ya)/(imgy-1)+ya
    for x in range (imgx):
        zx=x*(xb-xa)/(imgx-1)+xa
        z=complex(zx,zy)
        for i in range (maxit):
            z0=f(z)
            if abs(z)>1000:
                break
            z=z0
            r=i*139
            g=i*69
            b=i*35
            image.putpixel((x,y),(r,g,b))
image
```
![newton1](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/julia1.png)

#### *Ejemplo 2* - $f(z)= z+z^{2}-z^{3}$
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
 
#### *Ejemplo 3* - $f(z)= \sin(z^{3})$
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
 
#### *Ejemplo 4* - $f(z)= \cos(z)+z^{5}-\sin(z)$
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
 
     
