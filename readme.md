# Clasificación de Razas de Perros y Gatos mediante CNN

## Evaluación Parcial N.º 2 – TLY1101

**Asignatura:** Técnicas Avanzadas de Machine Learning I  
**Proyecto:** Clasificación multiclase de imágenes  
**Dataset:** Oxford-IIIT Pet Dataset  
**Tecnología:** Python, TensorFlow y Keras

---

## 1. Descripción del proyecto

El objetivo de este proyecto es desarrollar una Red Neuronal Convolucional (CNN) capaz de clasificar imágenes de perros y gatos en 37 razas diferentes, utilizando el conjunto de datos Oxford-IIIT Pet Dataset.

Este trabajo da continuidad a la Evaluación Parcial N.º 1, en la cual se implementó un modelo de clasificación basado en un Perceptrón Multicapa (MLP).

En esta segunda etapa se desarrolla una arquitectura CNN para comparar su desempeño con el modelo anterior y analizar las ventajas de utilizar redes convolucionales en problemas de clasificación de imágenes.

El proyecto considera la preparación de imágenes, entrenamiento de modelos, evaluación mediante métricas de clasificación, visualización de características y análisis de errores.

## 2. Objetivos

### Objetivo general

Desarrollar una Red Neuronal Convolucional para clasificar imágenes de mascotas en 37 razas diferentes y comparar su desempeño con el modelo MLP desarrollado anteriormente.

### Objetivos específicos

- Preparar las imágenes para su procesamiento mediante una CNN.
- Dividir los datos en conjuntos de entrenamiento, validación y prueba.
- Implementar una arquitectura CNN utilizando TensorFlow/Keras.
- Entrenar el modelo y analizar su comportamiento.
- Evaluar los modelos mediante distintas métricas de clasificación.
- Comparar el desempeño del MLP y la CNN.
- Visualizar Feature Maps y mapas de activación mediante Grad-CAM.
- Analizar las predicciones correctas e incorrectas.
- Identificar las clases que presentan mayores dificultades de clasificación.

## 3. Dataset utilizado

Se utiliza el **Oxford-IIIT Pet Dataset**, que contiene imágenes de perros y gatos correspondientes a 37 razas diferentes.

### Características principales

| Característica | Descripción |
|---|---|
| Dataset | Oxford-IIIT Pet Dataset |
| Cantidad de clases | 37 |
| Tipo de imágenes | RGB |
| Problema | Clasificación multiclase |
| Resolución utilizada | 64 × 64 píxeles |
| Formatos considerados | JPG, JPEG y PNG |

La variante seleccionada considera las 37 clases disponibles, manteniendo el mismo problema general trabajado en la Evaluación Parcial N.º 1.

## 4. Tecnologías y bibliotecas utilizadas

El proyecto fue desarrollado utilizando Python y las siguientes bibliotecas:

| Biblioteca | Utilización |
|---|---|
| TensorFlow | Construcción y entrenamiento de redes neuronales. |
| Keras | Implementación de las arquitecturas MLP y CNN. |
| NumPy | Operaciones y manipulación de datos numéricos. |
| Matplotlib | Visualización de imágenes, gráficos y resultados. |
| Scikit-learn | División de datos y métricas de clasificación. |
| OS / pathlib | Acceso y organización de archivos e imágenes. |
| Time | Medición del tiempo de entrenamiento. |

## 5. Preparación de los datos

Antes de entrenar los modelos se realiza un proceso de preparación de las imágenes.

### 5.1 Carga de imágenes

Se identifican automáticamente las carpetas correspondientes a las 37 clases y se obtienen las rutas de las imágenes disponibles.

### 5.2 Redimensionamiento

Todas las imágenes se redimensionan a 64 × 64 píxeles para mantener dimensiones uniformes.

### 5.3 Conversión RGB

Las imágenes se procesan utilizando tres canales de color: rojo, verde y azul.

### 5.4 Normalización

Los valores de los píxeles se dividen por 255 para transformarlos a un rango entre 0 y 1.

### 5.5 División de los datos

Se utiliza una división estratificada con la siguiente distribución:

| Conjunto | Porcentaje | Función |
|---|---:|---|
| Entrenamiento | 70 % | Aprendizaje del modelo. |
| Validación | 15 % | Seguimiento del aprendizaje. |
| Prueba | 15 % | Evaluación final. |

Se establece una semilla aleatoria de 42 para favorecer la reproducibilidad de la partición.

### 5.6 Organización de los datos

Se utiliza `tf.data.Dataset` para organizar las imágenes en lotes de 32 elementos.

También se utilizan las operaciones `shuffle`, `batch` y `prefetch` para organizar y optimizar el procesamiento de los datos.

## 6. Modelo MLP

Como modelo de referencia se utiliza el Perceptrón Multicapa desarrollado en la Evaluación Parcial N.º 1, adaptado a la misma partición de datos.

### Arquitectura MLP

| Capa | Configuración |
|---|---|
| Flatten | Conversión de imagen a vector. |
| Dense | 512 neuronas, ReLU. |
| Dropout | 0.3. |
| Dense | 256 neuronas, ReLU. |
| Dropout | 0.2. |
| Dense de salida | 37 neuronas, Softmax. |

El modelo MLP transforma la imagen en un vector unidimensional, por lo que no conserva directamente la estructura espacial de los píxeles.

## 7. Modelo CNN

Se implementa una Red Neuronal Convolucional mediante TensorFlow/Keras.

La arquitectura utiliza tres capas convolucionales para extraer características visuales de las imágenes y capas densas para realizar la clasificación final.

### Arquitectura CNN

| Capa | Configuración | Función |
|---|---|---|
| Input | 64 × 64 × 3 | Entrada de imágenes RGB. |
| Conv2D 1 | 32 filtros, 3 × 3, ReLU | Extracción de características iniciales. |
| MaxPooling2D 1 | 2 × 2 | Reducción espacial. |
| Conv2D 2 | 64 filtros, 3 × 3, ReLU | Extracción de patrones más complejos. |
| MaxPooling2D 2 | 2 × 2 | Reducción espacial. |
| Conv2D 3 | 128 filtros, 3 × 3, ReLU | Extracción de características de mayor complejidad. |
| MaxPooling2D 3 | 2 × 2 | Reducción espacial. |
| GlobalAveragePooling2D | Promedio espacial | Resumen de los mapas de características. |
| Dense | 128 neuronas, ReLU | Clasificación intermedia. |
| Dropout | 0.4 | Regularización. |
| Dense de salida | 37 neuronas, Softmax | Clasificación de las razas. |

### Justificación de la arquitectura

- **Conv2D:** permite detectar características visuales como bordes, texturas y patrones.
- **MaxPooling2D:** reduce las dimensiones espaciales y conserva las activaciones más relevantes.
- **GlobalAveragePooling2D:** resume los mapas de características antes de la clasificación.
- **Dense:** combina las características extraídas para determinar la clase.
- **Dropout:** reduce el riesgo de sobreajuste durante el entrenamiento.
- **Softmax:** entrega probabilidades para las 37 clases.

### Hiperparámetros

| Parámetro | Valor |
|---|---|
| Optimizador | Adam |
| Función de pérdida | Sparse Categorical Crossentropy |
| Métrica | Accuracy |
| Épocas | 15 |
| Batch size | 32 |
| Resolución | 64 × 64 × 3 |
| Número de clases | 37 |

## 8. Entrenamiento

Ambos modelos se entrenan utilizando el conjunto de entrenamiento y se supervisan mediante el conjunto de validación.

Durante el entrenamiento se registran:

- Accuracy de entrenamiento.
- Accuracy de validación.
- Loss de entrenamiento.
- Loss de validación.
- Tiempo de entrenamiento.

Se generan gráficos para observar la evolución del aprendizaje durante las 15 épocas.

Estos resultados permiten identificar el comportamiento de cada arquitectura y posibles señales de sobreajuste.

## 9. Evaluación de los modelos

Los modelos se evalúan utilizando el conjunto de prueba.

Se calculan las siguientes métricas:

| Métrica | Descripción |
|---|---|
| Accuracy | Proporción de predicciones correctas. |
| Precision | Proporción de predicciones positivas correctas por clase. |
| Recall | Proporción de ejemplos reales identificados correctamente. |
| F1-Score | Equilibrio entre Precision y Recall. |
| Matriz de confusión | Permite identificar aciertos y errores entre las clases. |

Para Precision, Recall y F1-Score se utiliza el promedio macro, dando el mismo peso a cada una de las 37 clases.

### Resultados comparativos

Completar esta tabla con los valores obtenidos al ejecutar el notebook:

| Métrica | MLP | CNN |
|---|---:|---:|
| Accuracy | Pendiente | Pendiente |
| Precision macro | Pendiente | Pendiente |
| Recall macro | Pendiente | Pendiente |
| F1-Score macro | Pendiente | Pendiente |
| Parámetros | Pendiente | Pendiente |
| Tiempo de entrenamiento | Pendiente | Pendiente |

## 10. Comparación entre MLP y CNN

La comparación considera el desempeño de clasificación, cantidad de parámetros, tiempo de entrenamiento y capacidad de generalización.

El MLP procesa las imágenes como vectores, mientras que la CNN conserva su estructura espacial y utiliza filtros convolucionales para extraer características automáticamente.

La comparación experimental permite determinar cuál de las arquitecturas presenta mejores resultados para este problema específico.

No se asume previamente que la CNN obtendrá mejores métricas: la conclusión debe basarse en los resultados obtenidos durante la evaluación.

## 11. Visualización de Feature Maps

Se implementa una visualización de los mapas de características generados por las capas convolucionales.

Se utilizan las capas `conv1`, `conv2` y `conv3` para observar cómo evoluciona la extracción de información visual.

Se muestran ocho filtros por capa.

La interpretación general es:

- **Primera capa:** identifica características iniciales, como bordes y texturas.
- **Segunda capa:** combina características para representar patrones más elaborados.
- **Tercera capa:** representa características visuales de mayor complejidad.

Estos mapas permiten comprender cómo la CNN procesa y transforma la información de las imágenes.

## 12. Mapas de activación Grad-CAM

Se utiliza Grad-CAM para visualizar las regiones de las imágenes que contribuyen a la clasificación realizada por la CNN.

El procedimiento utiliza las activaciones de la última capa convolucional y los gradientes asociados a la clase seleccionada.

Posteriormente, se genera un mapa de calor que se superpone sobre la imagen original.

Se contempla la generación de mapas de activación para las 37 clases utilizando imágenes del conjunto de prueba.

Esta técnica facilita la interpretación visual de las predicciones del modelo.

## 13. Análisis de errores

Se analizan ejemplos correctamente clasificados e incorrectamente clasificados por la CNN.

Además, se calcula el porcentaje de error por clase para identificar las razas que presentan mayores dificultades.

Entre las posibles causas de error se encuentran:

- Similitudes físicas entre distintas razas.
- Variaciones de iluminación.
- Diferentes posiciones de las mascotas.
- Fondos y elementos que pueden distraer al modelo.
- Imágenes con características poco definidas.
- Resolución limitada de las imágenes.

Este análisis permite identificar fortalezas y limitaciones de la arquitectura implementada.

## 14. Metodología CRISP-DM

El proyecto se organiza considerando las siguientes etapas:

1. **Comprensión del negocio:** definición del problema de clasificación de razas.
2. **Comprensión de los datos:** revisión del Oxford-IIIT Pet Dataset y sus 37 clases.
3. **Preparación de los datos:** carga, redimensionamiento, normalización y división de imágenes.
4. **Modelamiento:** implementación y entrenamiento de los modelos MLP y CNN.
5. **Evaluación:** análisis de métricas, matrices de confusión y errores.
6. **Despliegue:** almacenamiento de los modelos entrenados en formato `.keras` para su utilización posterior.

## 15. Consideraciones éticas y limitaciones

El modelo depende de la calidad y diversidad de las imágenes utilizadas durante el entrenamiento.

Por lo tanto, no se puede garantizar que obtenga los mismos resultados al procesar imágenes externas con condiciones diferentes a las del conjunto de datos.

Una clasificación incorrecta puede entregar información equivocada sobre la raza de una mascota. Por ello, el modelo debe considerarse una herramienta experimental de apoyo y no un sistema infalible.

Entre las principales limitaciones se encuentran la resolución de entrada, la cantidad de épocas y las posibles similitudes visuales entre razas.

Como mejoras futuras se podrían implementar técnicas de aumento de datos, ajustar los hiperparámetros, aumentar las épocas de entrenamiento y probar modelos preentrenados mediante transferencia de aprendizaje.

## 16. Conclusiones

El proyecto permite implementar una Red Neuronal Convolucional para resolver un problema de clasificación multiclase de imágenes utilizando el Oxford-IIIT Pet Dataset.

Se desarrolló un proceso de preparación de datos, entrenamiento y evaluación de modelos MLP y CNN.

La utilización de capas convolucionales permite extraer características visuales automáticamente, mientras que las visualizaciones de Feature Maps y Grad-CAM ayudan a interpretar el funcionamiento interno de la arquitectura.

La comparación entre ambos modelos debe fundamentarse en las métricas, los tiempos de entrenamiento y la capacidad de generalización observada.

Finalmente, el trabajo permite comprender las diferencias entre un Perceptrón Multicapa y una Red Neuronal Convolucional, aplicando técnicas de Machine Learning a un problema real de clasificación de imágenes.

## 17. Estructura del proyecto

```text
proyecto/
│
├── dataset/
│   └── oxford_procesado_todas/
│
├── notebooks/
│   └── evaluacion_parcial_2.ipynb
│
├── models/
│   ├── modelo_mlp.keras
│   └── modelo_cnn.keras
│
├── images/
│   ├── matrices_confusion/
│   ├── feature_maps/
│   ├── gradcam/
│   └── resultados/
│
└── README.md
```

Esta estructura permite organizar los datos, notebooks, modelos entrenados y evidencias visuales del proyecto.

## 18. Instalación y ejecución

### Requisitos

Instalar las bibliotecas necesarias:

```bash
pip install tensorflow numpy matplotlib scikit-learn pandas openpyxl
```

### Ejecución

1. Descargar o disponer del Oxford-IIIT Pet Dataset.
2. Ubicar las imágenes dentro de la carpeta `dataset/oxford_procesado_todas/`.
3. Abrir el notebook `evaluacion_parcial_2.ipynb`.
4. Ejecutar las celdas en orden.
5. Revisar las métricas, gráficos, Feature Maps y mapas Grad-CAM.
6. Verificar que los modelos entrenados se guarden correctamente en la carpeta `models/`.

## 19. Reproducibilidad

Para favorecer la reproducción de los resultados:

- Se utiliza una semilla aleatoria de 42 para la partición de los datos.
- Se mantiene la misma partición para comparar los modelos.
- Se utiliza una resolución uniforme de 64 × 64 píxeles.
- Se mantienen los mismos conjuntos de prueba y validación.
- Se registran las métricas y los tiempos de entrenamiento.

Se recomienda registrar las versiones de Python y las bibliotecas utilizadas para facilitar futuras ejecuciones.

---

**Proyecto desarrollado para la Evaluación Parcial N.º 2 – TLY1101.**