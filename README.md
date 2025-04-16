# 🧠 Detección de Alzheimer mediante Imágenes de Resonancia Magnética (MRI)

Este proyecto tiene como objetivo desarrollar modelos de inteligencia artificial capaces de clasificar diferentes etapas del Alzheimer a partir de imágenes cerebrales por resonancia magnética (MRI). Compara el rendimiento de un modelo de fine-tuning basado en **InceptionV3** con un modelo personalizado desarrollado desde cero.

---

## 🧬 ¿Qué es el Alzheimer?

El **Alzheimer** es una enfermedad neurodegenerativa progresiva que afecta la memoria, el pensamiento y el comportamiento. Es la causa más común de demencia entre las personas mayores. A medida que la enfermedad avanza, las células cerebrales mueren, lo que lleva a una disminución de las funciones cognitivas y, finalmente, a la pérdida de la independencia.

> 🔍 **Importancia de la detección temprana:**  
Detectar el Alzheimer en sus primeras etapas permite iniciar tratamientos que pueden ralentizar el progreso de la enfermedad, mejorar la calidad de vida y planificar el futuro tanto del paciente como de su familia.

---

## 📁 Base de Datos

La base de datos utilizada proviene del conjunto [**Alzheimer MRI Dataset (OASIS)**](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset), que contiene imágenes de resonancia magnética cerebral clasificadas en 4 categorías:

- **Non Impairment**
- **Very Mild Impairment**
- **Mild Impairment**
- **Moderate Impairment**

Las imágenes están organizadas en carpetas por clase y tienen una resolución adecuada para su uso en modelos de visión por computadora.

```text
Non Impairment         ~2.560 imágenes  
Very Mild Impairment    ~2.560 imágenes  
Mild Impairment         ~2.560 imágenes  
Moderate Impairment     ~2.560 imágenes
```

## 🧠 Modelos

- Modelo **InceptionV3** aplicando **fine-tuning** en sus capas finales y agregando una nueva cabeza de clasificación adaptada a las 4 clases del dataset. (22.860.772 params - 13.057.732 trainable)
- Modelo **CNN** con capas convolucionales, de pooling y fully connected. (3.353.428 params - 3.352.020 trainable)



## 📈 Resultados

| Modelo                        | **Pérdida (Loss)** | **Precisión (Accuracy)** | **Precisión (Precision)** | **Recall** | **Especificidad** | **F1-Score** |
|------------------------------|--------------------|---------------------------|----------------------------|------------|--------------------|--------------|
| **CNN Personalizado**        | **0.1508**         | **0.9609**                | **0.9609**                 | **0.9609** | **0.9870**         | **0.9609**   |
| Fine-tuning InceptionV3      | 0.4078             | 0.9022                    | 0.9022                     | 0.9022     | 0.9674             | 0.9022       |


## CNN Personalizado

![image](https://github.com/user-attachments/assets/284459a9-27ff-4a85-8fc8-66004b071640)


## Fine-Tuning

![image](https://github.com/user-attachments/assets/70702d10-b827-40a8-b5b0-a8b332e1e645)

