# RemooVI - Eliminador de Fondos con IA

<div align="center">
  
![RemooVI Banner](https://img.shields.io/badge/RemooVI-Background%20Remover-purple?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

Una aplicación web moderna para eliminar fondos de imágenes utilizando inteligencia artificial.

[Características](#características) • [Instalación](#instalación) • [Uso](#uso)

</div>

---
## Preview

**Visita el sitio:** > [https://remoovi.vercel.app](https://remoovi.vercel.app)



## Descripción

RemooVI es una herramienta web que permite eliminar el fondo de imágenes de manera automática y precisa utilizando IA. Con una interfaz moderna y elegante, los usuarios pueden procesar sus imágenes en segundos de forma completamente gratuita.

## Características

-  **Precisión profesional**: Utiliza algoritmos de IA avanzados para detección precisa de objetos
-  **Procesamiento rápido**: Elimina fondos en segundos
-  **Interfaz moderna**: Diseño glassmorphism con gradientes y animaciones suaves
-  **Totalmente responsive**: Funciona perfectamente en dispositivos móviles y desktop
-  **Drag & Drop**: Arrastra y suelta imágenes para procesarlas al instante
-  **Descarga HD**: Obtén tus imágenes procesadas en alta calidad
-  **Completamente gratis**: Sin límites ni marcas de agua

## Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web moderno y rápido para Python
- **Rembg**: Librería de IA para eliminación de fondos
- **Uvicorn**: Servidor ASGI de alto rendimiento

### Frontend
- **HTML5 & CSS3**: Estructura y estilos modernos
- **Tailwind CSS**: Framework de utilidades CSS
- **Vanilla JavaScript**: Lógica del cliente sin dependencias

## Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos

1. **Clona el repositorio**
```bash
git clone https://github.com/NotExer/RemooVI.git
cd RemooVI
```

2. **Crea un entorno virtual (recomendado)**
```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

## Uso

1. **Inicia el servidor backend**
```bash
python main.py
```
El servidor estará disponible en `http://localhost:8000`

2. **Abre el frontend**
- Abre el archivo `index.html` en tu navegador
- O utiliza un servidor local como Live Server en VS Code

3. **Procesa imágenes**
- Haz clic en "Cargar imagen" o arrastra y suelta una imagen
- Espera a que la IA procese la imagen
- Descarga el resultado sin fondo


## API Endpoints

### POST `/remove-bg`
Elimina el fondo de una imagen.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (imagen en formato JPG, PNG, o WEBP)

**Response:**
- Content-Type: `image/png`
- Body: Imagen procesada sin fondo


## Características de Diseño

- **Glassmorphism**: Efectos de vidrio translúcido
- **Gradientes animados**: Fondos con efectos de blur dinámicos
- **Animaciones suaves**: Transiciones fluidas en todas las interacciones
- **Feedback visual**: Indicadores de carga y estados de procesamiento
- **Vista previa con patrón de transparencia**: Visualización profesional de imágenes PNG
