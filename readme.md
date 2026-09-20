# 🧠 INFORME TÉCNICO AVANZADO: Implementación, Dinámica de Optimización y Auditoría de Perceptrones Multicapa (MLP) en Espacios de Alta Dimensionalidad

> **Asignatura:** Técnicas Avanzadas de Machine Learning I  
> **Fecha:** Septiembre de 2026  
> **Repositorio Oficial:** [proyectoIA (GitHub)](https://github.com/Kneza1/proyectoIA)  

---

## 📋 Tabla de Contenidos
1. [Descripción del Problema de Negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del Proyecto](#2-objetivos-del-proyecto)
3. [Definición de KPIs](#3-definición-de-kpis-indicadores-clave-de-desempeño)
4. [Fuentes de Datos y Espacios de Entrada](#4-fuentes-de-datos-y-espacios-de-entrada)
5. [Preparación, EDA y Transformación Tensorial](#5-preparación-eda-y-transformación-tensorial)
6. [Metodología CRISP-DM](#6-metodología-crisp-dm)
7. [Arquitectura del Modelo e Hiperparámetros](#7-arquitectura-del-modelo-e-hiperparámetros-especificaciones-técnicas)
8. [Evaluación del Desempeño y Convergencia](#8-evaluación-del-desempeño-y-análisis-de-convergencia)
9. [Auditoría de Errores y Limitaciones Teóricas del MLP](#9-auditoría-de-errores-y-limitaciones-teóricas-del-mlp)
10. [Conclusiones y Propuestas de Mejora Avanzada](#10-conclusiones-y-propuestas-de-mejora-avanzada)

---

## 1. 🏢 Descripción del Problema de Negocio
En el ecosistema industrial, agrícola, cartográfico y de accesibilidad actual, el procesamiento automatizado de flujos de datos visuales y tabulares de alta dimensionalidad representa un desafío crítico de ingeniería. Las revisiones humanas tradicionales en tareas como:
* La clasificación geoespacial (**Intel**).
* El diagnóstico fitosanitario de cultivos (**PlantVillage**).
* La taxonomía zoológica automatizada (**Oxford-IIIT**).
* La traducción de lenguajes gestuales (**Sign MNIST**).

Resultan operativamente ineficientes y propensas a sesgos subjetivos. El presente proyecto implementa un *framework* experimental basado en aprendizaje supervisado profundo para automatizar estos procesos, estableciendo una línea base (*baseline*) rigurosa mediante arquitecturas de Perceptrón Multicapa (MLP).

---

## 2. 🎯 Objetivos del Proyecto
* **Objetivo General:** Evaluar empírica y teóricamente el comportamiento de redes neuronales totalmente conectadas (MLP) frente a problemas complejos de clasificación en espacios de alta dimensionalidad (imágenes y tensores tabulares).
* **Objetivos Específicos:**
  1. Analizar el impacto de la transformación de tensores espaciales en vectores unidimensionales y su repercusión en la pérdida de topología.
  2. Implementar un pipeline reproducible bajo el ciclo CRISP-DM, gestionando hiperparámetros de regularización y optimización dinámica.
  3. Auditar cuantitativamente el rendimiento mediante métricas multiclase y binaria (*Accuracy*, *Precision*, *Recall*, *F1-Score* con promedios *Macro/Weighted*).
  4. Analizar la superficie de pérdida (*loss landscape*) y los indicios de sobreajuste a través de curvas de aprendizaje.
  5. Fundamentar técnicamente las limitaciones estructurales de las capas densas frente a arquitecturas convolucionales avanzadas.

---

## 3. 📈 Definición de KPIs (Indicadores Clave de Desempeño)
Para validar la viabilidad operativa de los modelos implementados:
* **Accuracy Global $\ge 80\%$:** Consistencia predictiva general en los subconjuntos de validación independientes.
* **Recall y F1-Score Críticos $\ge 85\%$:** Minimización de tasas de falsos negativos en dominios sensibles (como la detección de patologías agrícolas o triaje).
* **Eficiencia de Convergencia:** Minimización de la brecha de generalización (*generalization gap*) entre entrenamiento y validación.

---

## 4. 🗂️ Fuentes de Datos y Espacios de Entrada
El estudio abarcó cuatro dominios experimentales con distintas naturalezas tensoriales:

| Dominio / Escenario | Dataset Utilizado | Naturaleza del Input | Dimensión / Formato |
| :--- | :--- | :--- | :--- |
| **1. Cartografía** | Intel Image Classification | Imágenes RGB | $150 \times 150 \text{ px}$ (Directorios) |
| **2. Agricultura** | PlantVillage Dataset | Imágenes foliares | Directorios jerárquicos |
| **3. Mascotas** | Oxford-IIIT Pet Dataset | Taxonomía zoológica | Extracción automática de ficheros |
| **4. Accesibilidad** | Sign Language MNIST | Tensores tabulares | Vectores de 784 features (`.csv`) |

---

## 5. 🛠️ Preparación, EDA y Transformación Tensorial
Para mitigar la maldición de la dimensionalidad inherente a los MLP:
* **Reducción de Espacio Muestral:** Estandarización de resoluciones a $64\times64\times3$ y $28\times28\times1$ para acotar el número de pesos sinápticos iniciales.
* **Escalado Min-Max:** Transformación lineal de los tensores de píxeles al intervalo acotado $[0, 1]$ para estabilizar la varianza de las activaciones iniciales.
* **Partición Estratificada:** División 80/20 con fijación estricta de semilla (`seed=42`) para asegurar la determinística separación de los conjuntos de datos.

---

## 6. 🔄 Metodología CRISP-DM
El desarrollo se rigió bajo el estándar industrial *Cross-Industry Standard Process for Data Mining*:
1. **Comprensión del Negocio:** Modelado de variantes binarias y multiclase.
2. **Comprensión de Datos (EDA):** Auditoría de distribuciones y desbalance de clases.
3. **Preparación:** Normalización, aplanamiento y partición tensorial.
4. **Modelado:** Configuración secuencial de capas densas, Dropout y funciones de pérdida.
5. **Evaluación:** Auditoría con matrices de confusión y análisis de convergencia.
6. **Despliegue:** Exportación de artefactos binarios (`.keras`) y control de versiones en GitHub.

---

## 7. ⚙️ Arquitectura del Modelo e Hiperparámetros (Especificaciones Técnicas)
* **Capa de Entrada y Aplanamiento:** `Flatten` para transformar tensores espaciales en vectores de características $X \in \mathbb{R}^N$.
* **Capas Ocultas:** Estructuras en embudo con capas densas (`Dense`) de 512/256 y 128 neuronas, utilizando activaciones **ReLU** para mitigar el desvanecimiento del gradiente (*vanishing gradient*).
* **Regularización:** Inclusión de capas `Dropout` (0.2 y 0.3) para actuar como un ensamblaje implícito de subredes y evitar la coadaptación de pesos.
* **Capa de Salida:** Activación `sigmoid` (entropía cruzada binaria) para escenarios de dos clases, y `softmax` (entropía cruzada categórica dispersa) para escenarios multiclase.
* **Optimizador y Dinámica:** Algoritmo **Adam** por su ajuste adaptativo de la tasa de aprendizaje mediante estimación de momentos de primer y segundo orden.

---

## 8. 📊 Evaluación del Desempeño y Análisis de Convergencia
Los modelos fueron evaluados computacionalmente mediante:
* **Métricas de Rendimiento:** *Accuracy*, *Precision*, *Recall*, *F1-Score* (ponderados y por clase) para evaluar el impacto de posibles sesgos de clase.
* **Matrices de Confusión:** Visualización de la distribución de errores de predicción en espacios multiclase e identificación de clases solapadas.
* **Curvas de Aprendizaje:** Análisis de la evolución temporal de la función de pérdida (`Loss`) y precisión (`Accuracy`), identificando el punto óptimo de parada (*early stopping* implícito).

---

## 9. ⚠️ Auditoría de Errores y Limitaciones Teóricas del MLP
El análisis de ejemplos incorrectamente clasificados evidenció fallas en casos de frontera visual (iluminación, oclusiones y fondos complejos). Desde la perspectiva teórica de **Machine Learning Avanzado**, se documentaron las restricciones críticas del MLP en visión artificial:
1. **Destrucción de la Topología Espacial:** La pérdida de vecindad bidimensional al vectorizar los píxeles anula la correlación espacial local.
2. **Ausencia de Invarianza a la Traslación:** El perceptrón carece de invarianza geométrica; desplazar un objeto en el plano de la imagen altera radicalmente la activación de las neuronas.
3. **Costo Paramétrico Masivo:** El número de parámetros libres escala en función de $O(N \cdot M)$, lo que eleva exponencialmente el riesgo de sobreajuste y la dependencia de un volumen masivo de datos.

---

## 10. 🚀 Conclusiones y Propuestas de Mejora Avanzada
* **Conclusiones:** Se validó con éxito la ejecución del pipeline completo de Machine Learning bajo múltiples variantes experimentales. Los MLP demostraron utilidad en tareas acotadas o de bajo solapamiento visual, pero confirmaron su ineficiencia teórica ante la complejidad espacial de las imágenes.
* **Propuesta de Mejora Avanzada:** Se recomienda transicionar formalmente hacia **Redes Neuronales Convolucionales (CNN)**, aprovechando la extracción jerárquica de características mediante filtros locales compartidos (`Conv2D`) y la reducción de dimensionalidad espacial invariante (`MaxPooling2D`).