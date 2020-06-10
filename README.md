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
 
## ¿Qué es un fractal?:
Definido de forma sencilla y concisa, un fractal es una forma geométrica irregular o fragmentada que se puede dividir en partes, donde cada una de las cuales son aproximadamente una copia de tamaño reducido del conjunto entero...

# Fractales de Newton, Raíces complejas

## ***Definición*** 
Cuando la búsqueda de la solución de un problema de aplicación implica la solución de ecuaciones no lineales se hace uso de métodos numéricos, comoo el método de Newton que es uno de los más usados debido a su versatilidad y agilidad para aprosimar las soluciones de ecuaciones no lineales. Solucionar ecuaciones con variable compleja a través del método de Newton tiene una aplicación muy interesante en el campo de los fractales como es la del problema de Cayley y las figuras fractales que se producen a partir de la convergencia, divergencia e incluso la eficiencia del método. En este artículo se muestra el estudio del problema de Cayley a través de la generalización del método de Newton del $\mathbb R^{2}$. Además, se presentan algunos fractales producidos por iteraciones del método de Newton en los complejos.
  
### ***Algunos Fractales*** 
 Los siguientes ejemplos de fractales son realizados con código Python, donde se relacionan funciones polonomiales y/o trigonométricas.
 
**-** los paquetes necesarios para la implementación de los códigos de los fractales son:
 ```
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
 ```
 
#### *Ejemplo 1* - $f(z)= z^{3}+z^{2}$

![newton1](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton1.png)


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

#### *Ejemplo 2* - $f(z)= z+z^{2}-z^{3}$

 ![newton2](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton2.png)

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

 
#### *Ejemplo 3* - $f(z)= \sin(z^{3})$

 ![newton3](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton3.png)

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

 
#### *Ejemplo 4* - $f(z)= \cos(z)+z^{5}-\sin(z)$

 ![newton4](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton4.png)
 
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
 
#### *Ejemplo 5* - $f(z)= (z^{7}-z^{5})*\sin(z)$

 ![newton5](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/Newton5.png)

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
 

# Conjuntos De Julia
 
## ***Definición*** 
Otro metodo para construir fractales son los conjuntos de Julia, así llamados por el matemático Gaston Julia, donde se forman familias de conjuntos fractale, como resultado del estudio y análisis del comportamiento de los números complejos al ser iterados por una función.
Una familia muy importante en conjuntos de Julia se obtiene a partir de funciones cuadráticas simples,  de la forma $F(z) = z^{2} + c$  , donde $c$  es un número complejo. Como por ejemplo:

$$f(z)=z^{2}+c$$, donde $$c=-0.8,+0.156i$$

![juliaejemplo](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/ejemplojulia.png)

El conjunto de Julia que se obtiene a partir de esta función se denota **$Jc$**. El proceso para  obtener este conjunto de Julia de  es el siguiente:
Se elige un número complejo cualquiera $z$ y se va construyendo una sucesión de números de la siguiente manera:

$$z_{0} = z$$
$$z_{1} = F(z_{0})= z_{02} + c$$
$$z_{2} = F(z_{1}) =z_{12}+c$$
$$z_{n+1} =  F(z_{n}) =z_{n2}+c$$

Si esta sucesión queda acotada, entonces se dice que $z$ pertenece al conjunto de Julia de parámetro $c$, denotado por **$Jc$**; de lo contrario, si la sucesión tiende al infinito, $z$ queda excluido de éste. Es fácil deducir que obtener un conjunto de Julia resulta muy laborioso, pues el proceso anterior habría que repetirlo para cualquier número complejo $z$, e ir decidiendo en cada caso si dicho número pertenece o no al conjunto **$Jc$**. Debido a la infinidad de cálculos que se necesitaban  para  obtener la gráfica correspondiente, Pero solo hasta los años 80s gracias a todos los avances computacionales se logro porfin verlos en una pantalla, lastimosamente Gaston Julia no alcanzo a verlo por si mismo:

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

#### Triangulo de Sierpinski,
el cual, sin importar con que figura se implemente el atgoritmo se llegara a la misma transformación compuesta por triangulos. costruimos el algoritmo mediante código Python
#### Iteración 0
 ![rianguloSierpinski](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/trianguloSierpinski.png)
#### Iteración 7
![rianguloSierpinski6](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/trianguloSierpinski6.png)
```
import numpy as np
import matplotlib.pylab as plt

def SierpinskiTriangle(a, b, c, iterations):
    if iterations == 0:
        plt.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]], 'g') 
        plt.draw()
    else: 
        SierpinskiTriangle(a, (a + b) / 2., (a + c) / 2., iterations - 1) 
        SierpinskiTriangle(b, (b + a) / 2., (b + c) / 2., iterations - 1) 
        SierpinskiTriangle(c, (c + a) / 2., (c + b) / 2., iterations - 1)
        
a = np.array([0, 0])
b = np.array([1, 0])
c = np.array([0.5, np.sqrt(3)/2.])

iterations = 0

fig = plt.figure(figsize=(8,8))

SierpinskiTriangle(a, b, c, iterations)

plt.axis('equal')
plt.axis('off')


iterations = 6

plt.figure(figsize=(8,8))

SierpinskiTriangle(a, b, c, iterations)

plt.axis('equal')
plt.axis('off')
plt.show()
```

**-** *Otro ejemplo de algoritmo de determinita, es otro de los algoritmos de sierpinski para un cuadro, mas conocido como "la alfombra de Sierpinski"*
#### Iteración 0
 ![alfombraSierpinski](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/alfombra.png)
#### Iteración 4
![alfombraSierpinski4](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/alfombra4.png)
```
import numpy as np
import matplotlib.pylab as plt

def carpet(a, b, c, d, iterations, offset=np.array([0,0])):
    ab = (a + b)/3.
    ba = 2*(a + b)/3.
    bc = (2*b + c)/3.
    cb = (b + 2*c)/3.
    dc = (c + 2*d)/3.
    cd = (2*c + d)/3.
    ad = (d + a)/3.
    da = 2*(d + a)/3.
    abd = 2*a/3. + (b + d)/3.
    bac = a + (2*b + d)/3.
    cbd = 4*a/3. + 2*(b + d)/3.
    dac = a + (b + 2*d)/3.
    plt.fill([abd[0]+offset[0], bac[0]+offset[0],
    cbd[0]+offset[0], dac[0]+offset[0]],
                 [abd[1]+offset[1], bac[1]+offset[1],
    cbd[1]+offset[1], dac[1]+offset[1]], 'blueviolet')
    plt.hot()
    if iterations == 0:
        plt.fill([abd[0]+offset[0], bac[0]+offset[0],
        cbd[0]+offset[0], dac[0]+offset[0]],
                 [abd[1]+offset[1], bac[1]+offset[1],
            cbd[1]+offset[1], dac[1]+offset[1]], 'blueviolet')
        plt.hot()
    else:
        #paso 1
        a_m =np.array([0,0])
        ab_m = ab - a
        abd_m = abd - a
        ad_m = ad - a
        offset1= offset +a
        carpet(a_m, ab_m, abd_m, ad_m, iterations - 1,offset1)
        #paso 2
        ab_m =np.array([0,0])
        ba_m = ba - ab
        bac_m = bac - ab
        abd_m = abd - ab
        offset2= offset +ab
        carpet(ab_m, ba_m, bac_m, abd_m, iterations - 1,offset2)
        #paso 3
        ba_m =np.array([0,0])
        b_m = b - ba
        bc_m = bc - ba
        bac_m = bac - ba
        offset3= offset +ba
        carpet(ba_m, b_m, bc_m, bac_m, iterations - 1,offset3)
        #paso 4
        bac_m =np.array([0,0])
        bc_m = bc - bac
        cb_m = cb - bac
        cbd_m = cbd - bac
        offset4= offset +bac
        carpet(bac_m, bc_m, cb_m, cbd_m, iterations - 1,offset4)
        #paso 5
        cbd_m = np.array([0, 0])
        cb_m = cb - cbd
        c_m = c - cbd
        cd_m = cd - cbd
        offset5= offset +cbd
        carpet(cbd_m, cb_m, c_m, cd_m, iterations - 1,offset5)        
        #paso 6
        dac_m = np.array([0, 0])
        cbd_m = cbd - dac
        cd_m = cd - dac
        dc_m = dc - dac
        offset6= offset +dac
        carpet(dac_m, cbd_m, cd_m, dc_m, iterations - 1,offset6)     
        #paso 7
        da_m = np.array([0, 0])
        dac_m = dac - da
        dc_m = dc - da
        d_m = d - da
        offset7= offset +da
        carpet(da_m, dac_m, dc_m, d_m, iterations - 1,offset7)        
        #paso 8
        ad_m = np.array([0, 0])
        abd_m = abd - ad
        dac_m = dac - ad
        da_m = da - ad
        offset8= offset +ad
        carpet(ad_m, abd_m, dac_m, da_m, iterations - 1,offset8)

a = np.array([0, 0])
b = np.array([3, 0])
c = np.array([3, 3])
d = np.array([0, 3])
ab = (a + b)/3.
ba = 2*(a + b)/3.
bc = (2*b + c)/3.
cb = (b + 2*c)/3.
dc = (c + 2*d)/3.
cd = (2*c + d)/3.
ad = (d + a)/3.
da = 2*(d + a)/3.

fig = plt.figure(figsize=(10,10))

iterations = 0

carpet(a, b, c, d, iterations)
plt.plot([a[0],b[0],c[0],d[0],a[0]],[a[1],b[1],c[1],d[1],a[1]],'k-',lw=3)
plt.plot([ab[0],dc[0]],[ab[1],dc[1]],'k--',lw=3)
plt.plot([ba[0],cd[0]],[ba[1],cd[1]],'k--',lw=3)
plt.plot([ad[0],bc[0]],[ad[1],bc[1]],'k--',lw=3)
plt.plot([da[0],cb[0]],[da[1],cb[1]],'k--',lw=3)
plt.axis('equal')
plt.axis('off')

iterations = 4

plt.figure(figsize=(10,10))
carpet(a, b, c, d, iterations)
plt.axis('equal')
plt.axis('off')

plt.show()
```

**-** El algoritmo aleatorio es similar, pero en lugar de aplicar las funciones a un conjunto de puntos, las aplicamos sobre un único punto, que vamos dibujando. A cada una de las transformaciones del sistema le asignamos un valor de probabilidad, teniendo en cuenta que la suma total de los valores de probabilidad de las funciones debe valer 1. En cada iteración del algoritmo, seleccionamos una de las transformaciones con probabilidad p. Esto es muy sencillo de hacer, simplemente se obtiene un valor aleatorio entre 0 y 1, por ejemplo la conocida hoja de helecho, para su representacion se utiliza el siguiente codigo Python:

![helecho](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/helecho.png)

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

**-** Sierpinski tambien desarrolló un algoritmo aletorio que transforma, o mejor dicho fragmenta figuras geometricas en diferentes triangulos al azar, como por ejemplo:
#### Para un cuadro se tiene:
![cuadroaleatorio](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/cuadrosierpinski.png)
```
import numpy as np
import matplotlib.pylab as plt
import random

def bis(x,y,k):
    return (y - x)/float(k) + x

def f(a,b,c,k):
    return bis(a,b,k), bis(b,c,k), bis(a,c,k)

def Sierpinski(a,b,c,k,iteration): 
    
    x=(random.random(),random.random(),random.random())
    if iteration==0:
        plt.fill([a[0], b[0], c[0]], [a[1], b[1], c[1]],color=x,alpha=0.9)
        plt.hot()
        
    else:
        Sierpinski(a,bis(a,b,k),bis(a,c,k),k,iteration-1)
        Sierpinski(b,bis(a,b,k),bis(b,c,k),k,iteration-1)
        Sierpinski(c,bis(a,c,k),bis(b,c,k),k,iteration-1)
        plt.hot()
        
        plt.fill([bis(a,b,k)[0], bis(a,c,k)[0], bis(b,c,k)[0]],
[bis(a,b,k)[1], bis(a,c,k)[1], bis(b,c,k)[1]], color=x,alpha=0.9)

h = np.sqrt(3)

a1 = np.array([0,0])
b1 = np.array([3,0])
c1 = np.array([1.5,h])
a1u = np.array([0,2*h])
b1u = np.array([3,2*h])
c1u = np.array([1.5,h])

a2 = np.array([0,0])
b2 = np.array([0,2*h])
c2 = np.array([1.5,h])
a2r = np.array([3,0])
b2r = np.array([3,2*h])
c2r = np.array([1.5,h])

k1 = 3
k1u = 5
k2 = 4
k2r = 6

fig, ax = plt.subplots(1,figsize=(15,15)) 

Sierpinski(a1,b1,c1,k1,iteration=7) 
plt.hot()
Sierpinski(a1u,b1u,c1u,k1u,iteration=7) 
plt.hot()
Sierpinski(a2,b2,c2,k2,iteration=7) 
plt.hot()
Sierpinski(a2r,b2r,c2r,k2r,iteration=7) 
plt.hot()

ax.set_xlim(0,1) 
ax.set_ylim(0,1) 
plt.axis('equal')
plt.axis('off')
plt.show()
```
#### Para una estrella se tiene:
![estrellaleatorio](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/triangulosierpinskirandom.png)
```
import numpy as np
import matplotlib.pyplot as plt
import random

def gasket(pa, pb, pc, level):
    x=(random.random(),random.random(),random.random())
    if level == 0:
        plt.fill([pa[0], pb[0], pc[0]], [pa[1], pb[1], pc[1]], color=x,alpha=0.9) 
        plt.hot()
    else:
        gasket(pa, (pa + pb) / 2., (pa + pc) / 2., level - 1) 
        gasket(pb, (pb + pa) / 2., (pb + pc) / 2., level - 1) 
        gasket(pc, (pc + pa) / 2., (pc + pb) / 2., level - 1)
        plt.hot()
        plt.fill([(pa[0] + pb[0]) / 2.,(pb[0] + pc[0]) / 2.,(pa[0] + pc[0]) / 2.],
                 [(pa[1] + pb[1]) / 2.,(pb[1] + pc[1]) / 2.,(pa[1] + pc[1]) / 2.],color=x,alpha=0.9)

A = np.array([0,28]) 
B = np.array([29,7])
C = np.array([18.5,-27])
D = np.array([-17.5,-27.5]) 
E = np.array([-29,6.5]) 
L = np.array([-7,6.5])
K = np.array([7,7])
M = np.array([11.5,-6])
N = np.array([0.5,-14.5])
O = np.array([-11,-6.5])
origin = np.array([0,-3])

level = 5
fig, ax = plt.subplots(1,figsize=(15,15)) 
gasket(A, L, K, level) 
gasket(B, K, M, level)
gasket(C, M, N, level)
gasket(D, N, O, level)
gasket(E, O, L, level)
gasket(origin, L, K, level)
gasket(origin, K, M, level)
gasket(origin, M, N, level)
gasket(origin, N, O, level)
gasket(origin, O, L, level)
plt.hot()
ax.set_xlim(0,1.2) 
ax.set_ylim(0,1.2) 
plt.axis('equal')
plt.axis('off')
plt.show()
```

# Factal 3D
Gracias también a los avances computacionales, hoy podemos disfrutar de paisajes matemáticos, tales como los fractales 3D, un ejemplo de estos puede ser el siguiente contorno de un fractal de Mandelbrot o también denominado como una parcela de Mandelbrot, realizado con código Python:
![fractal3d](https://raw.githubusercontent.com/MiguelACC202/Galeria-De-Fractales/master/fractal3d.png)
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








# Cibergrafía
**1.** Graphene Themes, Aprendemos Matemáticas.CONJUNTOS DE JULIA Y MANDELBROT. Recuperado el 6 de septiembre de 2020, disponible en:http://www3.gobiernodecanarias.org/medusa/ecoblog/mrodperv/fractales/conjuntos-de-julia-y-mandelbrot/f.

**2.** J. Terán Tarapués, C. Rúa Alvarez (2018). WEL MÉTODO DE NEWTON PARA RAÍCES COMPLEJAS. FRACTALES EN EL PROBLEMA DE CAYLEY (Revista EIA, vol. 15, núm. 29, 2018). [En línea]. Recuperado el 6 de septiembre de 2020. Disponible en: https://www.redalyc.org/jatsRepo/1492/149256546008/html/index.html 

**3.** J. Llopis. Sistema de Funciones Iteradas: Fractales Autosemejantes. Recuperado el 7 de septiembre de 2020, disponible en: https://www.matesfacil.com/fractales/autosemejantes/fractal-autosemejante-definicion-ejemplos-funcion-propiedades-teorema-punto-fijo-compacto-iteraciones-sistema-contractivas-iteradas-galeria.html.

**4.** M. Díaz Kusztricn (20 de enero de 2017).3Dibujar fractales con Sistemas de Funciones Iteradas (IFS). Recuperado el 8 de septiembre de 2020, disponible en: http://software-tecnico-libre.es/es/articulo-por-tema/todas-las-secciones/todos-los-temas/todos-los-articulos/dibujar-con-sistemas-de-funciones-iteradas.

**5.** DIVULGARE (23 mayo, 2011).Fractales en 3D. Recuperado el 9 de septiembre de 2020, disponible en: http://www.divulgare.net/fractales-en-3d/.

**6.** A. Strumia (2011).3D fractal landscapes, Rendering structures as wholes or by sequential or randomprocesses. Recuperado el 9 de septiembre de 2020, disponible en:http://inters.org/files/research/forminfo/04_chapter_04.pdf.
