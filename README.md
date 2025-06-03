# 📉 **Regularización en Regresión: Domina el Sesgo-Varianza**

🔬 *“La simplicidad no es una meta. Es el subproducto de una buena idea y expectativas modestas.” – Paul Rand*

Source: [scikit-learn](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

## 🚀  **Métodos de regularización**

1. **Ridge Regression (L2)**

Para la estimación de los coeficientes en mínimos cuadrados debemos minimizar la suma de los errores al cuadrado. Para generar una regresión tipo rigde agregamos la penalización y mínimizamos la expresión:

$$\begin{equation}
\sum_{i=1}^n{(y_i - \beta_o - \sum_{j=1}^p{\beta_jx_{ij}})^2} + \lambda\sum_{j=1}^p{\beta_{j}^2}
\end{equation}$$

Donde la primera expresión es la suma de los errores al cuadrado y $$\lambda$$ es un parámetro que debe ser tuneado.

2. **Lasso Regression (L1)**

A diferencia de Ridge, matemáticamente el único cambio es que ahora los coeficientes de la penalización están en valor absoluto en vez de elevados al cuadrado. Esto tiene efectos distintos a la Ridge:  

- No penaliza de la misma manera a los coeficientes muy grandes.
- En la regresión Ridge los coeficientes tienden hacia cero, en la regresión Lasso los coeficientes puede volverse cero, lo que implica que la regresión Lasso tiene otro efecto y es que automáticamente depura las variables que no agregan poder predictivo al modelo.

$$\begin{equation}
\sum_{i=1}^n{(y_i - \beta_o - \sum_{j=1}^p{\beta_jx_{ij}})^2} + \lambda\sum_{j=1}^p{|\beta_{j}|}
\end{equation}$$

3. **Elastic Net**

Es una combinación de Ridge y Lasso. Se decide entonces qué peso se le da a cada método de penalización y se implementa la regresión:

$$\begin{equation}
\sum_{i=1}^n{(y_i - \beta_o - \sum_{j=1}^p{\beta_jx_{ij}})^2} + \lambda_{1}\sum_{j=1}^p{\beta_{j}^2} +\lambda_{2}\sum_{j=1}^p{|\beta_{j}|}
\end{equation}$$


## 🎯 **¿Por qué es importante?**

En un mundo lleno de datos, más variables no siempre significan mejores modelos. La regularización es una de las herramientas más poderosas para:

- Controlar el sobreajuste
- Mejorar la interpretabilidad del modelo
- Aumentar la robustez predictiva
- Seleccionar variables automáticamente (Lasso)
