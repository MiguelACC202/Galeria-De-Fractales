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
 Los siguientes ejemplos de fractales son realizados con código Python, donde se relacionan funciones polonomiales y/o trigonométricas.
 
**-** los paquetes necesarios para la implementación de los códigos de los fractales son:
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

Una familia muy importante de conjuntos de Julia se obtiene a partir de funciones cuadráticas simples,como por ejemplo: $Fc(z) = z^{2} + c$  , donde $c$  es un número complejo. Como por ejemplo:

$$f(z)=z^{2}+c$$, donde $$c=-0.8,+0.156i$$

![juliaejemplo](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/ejemplojulia.png)

El conjunto de Julia que se obtiene a partir de esta función se denota **$Jc$**. El proceso para  obtener este conjunto de Julia de  es el siguiente:
Se elige un número complejo cualquiera z y se va construyendo una sucesión de números de la siguiente manera:

$$z_{0} = z$$
$$z_{1} = F(z_{0})= z_{02} + c$$
$$z_{2} = F(z_{1}) =z_{12}+c$$
$$z_{n+1} =  F(z_{n}) =z_{n2}+c$$

Si esta sucesión queda acotada, entonces se dice que z pertenece al conjunto de Julia de parámetro $c$, denotado por **$Jc$**; de lo contrario, si la sucesión tiende al infinito, z queda excluido de éste. Es fácil deducir que obtener un conjunto de Julia resulta muy laborioso, pues el proceso anterior habría que repetirlo para cualquier número complejo z, e ir decidiendo en cada caso si dicho número pertenece o no al conjunto **$Jc$**. Debido a la infinidad de cálculos que se necesitaban  para  obtener la gráfica correspondiente, se tuvo que esperar hasta los años ochenta para poder  representar estos conjuntos. Pero gracias a todos los avances computacionales se logro porfin verlos en una pantalla, lastimosamente Gaston Julia no alcanzo a verlo por si mismo:
### ***Algunos Fractales*** 
 Los siguientes ejemplos de fractales son realizados con código Python, donde se relacionan funciones polonomiales y/o trigonométricas.

**-** los paquetes necesarios para la implementación de los códigos de los fractales son:
 ```
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
 ```
 
#### *Ejemplo 1* - $f(z)= z^{5}+c$, donde $c=-1-i$
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
![julia1](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/julia1.png)

#### *Ejemplo 2* - $f(z)= \sin(z^{3})+c$, donde $c=-1-i$
 ```
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-3
xb=3
ya=-3
yb=3
maxit=30
def f(z):
    return np.sin(z**3)+complex(-1,-1)
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
            r=i*25
            g=i*25
            b=i*112
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![julia2](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/julia2.png)
 
#### *Ejemplo 3* - $f(z)= \tan(z^{3})-5z^{5}+c$, donde $c=0.2+0.5i$
 ```
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-1
xb=1
ya=-1
yb=1
maxit=30
def f(z):
    return np.tan(z**3)-5*z**5+complex(0.2,0.5)
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
            r=i*220
            g=i*20
            b=i*60
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![julia3](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/julia3.png)
 
#### *Ejemplo 4* - $f(z)= z^{3}\sin(z)-z^{2}\cos(z)+zc$, donde $c=0.2+0.5i$
 ```
imgx=800
imgy=800
image=Image.new("RGB",(imgx,imgy))
xa=-5
xb=5
ya=-5
yb=5
maxit=30
def f(z):
    return (np.sin(z))*z**3-(np.cos(z))*z**2+z*complex(np.tan(0.6),0.36)
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
            r=i*255
            g=i*99
            b=i*71
            image.putpixel((x,y),(r,g,b))
image
 ```
 ![julia4](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/julia4.png)
 
## Sistemas de funciones iteradas
Los sistemas de funciones iteradas son conjuntos de n transformaciones afines contractivas. Normalmente se utilizan dos tipos de algoritmos, el algoritmo determinista y el algoritmo aleatorio.

**-** El algoritmo determinista trata de tomar un conjunto de puntos, de cualquier figura geométrica, y aplicarle cada una de las $n$ transformaciones afines del sistema, con lo cual obtenemos $n$ conjuntos de puntos transformados. A cada uno de ellos le volvemos a aplicar cada una de las $n$ funciones, obteniendo $n^{2}$ nuevos conjuntos de puntos. como por ejemplo:

#### Triangulo de sierpinski,
el cual, sin importar con que figura se implemente el atgoritmo se llegara a la misma transformación compuesta por triangulos. costruimos el algoritmo mediante el siguiente código Python
```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0,0],[1,0],[0,1],[0,0]])
plt.scatter(Tri.transpose()[0],Tri.transpose()[1])
plt.plot(Tri.transpose()[0],Tri.transpose()[1])
ax.set_xticks(np.arange(-0.2,1.4,0.2))
ax.set_yticks(np.arange(-0.2,1.4,0.2))
plt.grid()
ax.axis("equal")
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0,0]])
for i in range(8):
    tritrans=np.array([transafin([[0.5,0],[0,0.5]],[0,0],i) for i in Tri])
    tritrans2=np.array([transafin([[0.5,0],[0,0.5]],[0,0.5],i) for i in Tri])
    tritrans3=np.array([transafin([[0.5,0],[0,0.5]],[0.5,0],i) for i in Tri])
    Tri=np.concatenate((tritrans,tritrans2,tritrans3))
plt.scatter(Tri.transpose()[0],Tri.transpose()[1],color='black',s=0.2)
ax.set_xticks(np.arange(-0.2,1.4,0.2))
ax.set_yticks(np.arange(-0.2,1.4,0.2))
plt.grid()
ax.axis("equal")
```
![triangulo](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/triangulo.png)

**-** *De la misma forma para un conjunto de cuadros con una tranformación adicional en su algoritmo, da como resultado el siguiente SIF determinista*
```
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
for i in range(4):
  tritrans=np.array([transafin([[0.5,0],[0,0.5]],[0,0],i) for i in Tri])
  tritrans2=np.array([transafin([[0.5,0],[0,0.5]],[0,0.5],i) for i in Tri])
  tritrans3=np.array([transafin([[0.5,0],[0,0.5]],[0.5,0],i) for i in Tri])
  tritrans4=np.array([transafin([[0.5,0],[0,0.5]],[0.5,0],i) for i in Tri])
  Tri=np.concatenate((tritrans,tritrans2,tritrans3,tritrans4))
plt.scatter(tritrans.transpose()[0],tritrans.transpose()[1],color='g', s=0.1)
plt.scatter(tritrans2.transpose()[0],tritrans2.transpose()[1],color='r', s=0.1)
plt.scatter(tritrans3.transpose()[0],tritrans3.transpose()[1],color='b', s=0.1)
ax.set_xticks(np.arange(-0.2,1.4,0.2))
ax.set_yticks(np.arange(-0.2,1.4,0.2))
plt.grid()
ax.axis("equal")
```
![determinista](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/deterministico.png)

**-** El algoritmo aleatorio es similar, pero en lugar de aplicar las funciones a un conjunto de puntos, las aplicamos sobre un único punto, que vamos dibujando. A cada una de las transformaciones del sistema le asignamos un valor de probabilidad, teniendo en cuenta que la suma total de los valores de probabilidad de las funciones debe valer 1. En cada iteración del algoritmo, seleccionamos una de las transformaciones con probabilidad p. Esto es muy sencillo de hacer, simplemente se obtiene un valor aleatorio entre 0 y 1, por ejemplo la conocida hoja de helecho, para su representacion se utiliza el siguiente codigo Python:
```
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0.8,0.8]])
for i in range(8):
    tritrans=np.array([transafin([[0,0],[0,0.16]],[0,0],i) for i in Tri])
    tritrans2=np.array([transafin([[0.85,0.04],[-0.04,0.85]],[0,1.60],i) for i in Tri])
    tritrans3=np.array([transafin([[0.2,0-0.26],[0.23,0.22]],[0,1.60],i) for i in Tri])
    tritrans4=np.array([transafin([[-0.15,0.28],[0.26,0.24]],[0,0.44],i) for i in Tri])
    Tri=np.concatenate((tritrans,tritrans2,tritrans3,tritrans4))
plt.scatter(Tri.transpose()[0],Tri.transpose()[1],color='g',s=0.2)
plt.grid()
ax.axis("equal")
```
![helecho](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/helecho.png)
**-** Al igual que la siguiente transformación, donde en lugar de iterar un único punto, se itera diferentes puntos bajo un mismo parametro de algoritmo aleatorio:
```
fig=plt.figure()
ax=plt.gca()
Tri=np.array([[0,0],[0.5,0],[0.5,0.5],[0,1],[0,0]])
for i in range(8):
    tritrans=np.array([transafin([[0,0],[0,(1/3)]],[0,0],i) for i in Tri])
    tritrans2=np.array([transafin([[0,(2/3)],[(1/3),0]],[0,(1/3)],i) for i in Tri])
    tritrans3=np.array([transafin([[(1/3),(2/3)],[(2/3),0]],[(1/3),0],i) for i in Tri])
    tritrans4=np.array([transafin([[(2/3),(1/3)],[(2/3),(2/3)]],[(1/3),(1/3)],i) for i in Tri])
    Tri=np.concatenate((tritrans,tritrans2,tritrans3,tritrans4))
plt.scatter(tritrans.transpose()[0],tritrans.transpose()[1],color='g', s=0.1)
plt.scatter(tritrans2.transpose()[0],tritrans2.transpose()[1],color='r', s=0.1)
plt.scatter(tritrans3.transpose()[0],tritrans3.transpose()[1],color='b', s=0.1)
plt.scatter(tritrans4.transpose()[0],tritrans4.transpose()[1],color='y', s=0.1)
ax.set_xticks(np.arange(-0.5,6,0.5))
ax.set_yticks(np.arange(-0.5,6,0.5))
plt.grid()
ax.axis("equal")
```
![aleatorio](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/aleatorio.png)

# Factal 3D
Gracias también a los avances computacionales, hoy podemos disfrutar de paisajes matemáticos, tales como los fractales 3D, un ejemplo de estos puede ser el siguiente contorno de un fractal de Mandelbrot o también denominado como una parcela de Mandelbrot, realizado con el siguiente código Python:
```
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure as fg
from matplotlib import cm
import numpy as np 

fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=120,elev=45) 

ax.dist = 5 
ax.set_facecolor([0.05,0.05,0.05]) 
n = 16 
dx = -0.6 
dy = 0.03 
L = 1.3 
Max = 200

def f(Z):
    return np.e**(-np.abs(Z))

x = np.linspace(-L+dx,L+dx,M)
y = np.linspace(-L+dy,L+dy,M)
X,Y = np.meshgrid(x,y)
Z = np.zeros(Max)
W = np.zeros((Max,Max))
C = X + 1j*Y 

for k in range(1,n+1):
    ZZ = Z**2 + C
    Z = ZZ
    W = f(Z)
   
ax.set_xlim(dx-L,dx+L) 
ax.set_zlim(dy-L,dy+L) 
ax.set_zlim(-0.2*L,1.35*L)  
ax.contourf3D(X, Y, W, 2*n, cmap="magma") 
ax.axis("off")
plt.show()
```
![fractal3d](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/fractal3d.png)
