# Guía de Despliegue en Render (Paso a Paso)

Este documento explica cómo subir la aplicación completa a [Render](https://render.com) en su capa gratuita (**Free Tier**), para disponer de una versión funcional en línea accesible desde cualquier navegador.

---

## Arquitectura en la nube

El proyecto se despliega como dos servicios independientes dentro de la misma cuenta de Render:

1. **Backend (`tp-mesas-backend`):** Servicio Web en Python (FastAPI + Uvicorn) que sirve la API REST, ejecuta inferencias de Machine Learning y colorea el grafo con Welsh-Powell.
2. **Frontend (`tp-mesas-frontend`):** Sitio Estático (React 19 + TypeScript + Vite) que consume la API del backend y renderiza la interfaz interactiva.

---

## Opción A — Despliegue Automático con Blueprint (`render.yaml`) *(Recomendado)*

Render incluye soporte nativo para archivos de especificación (`render.yaml` o *Blueprints*), lo que permite levantar ambos servicios con un solo clic.

### Pasos:

1. Ingresar a [dashboard.render.com](https://dashboard.render.com) e iniciar sesión con tu cuenta de GitHub.
2. Hacer clic en el botón **New +** (arriba a la derecha) y seleccionar **Blueprint**.
3. Conectar y seleccionar el repositorio `TP-Aviani-Canal-Planificaci-n-de-mesas-de-examen`.
4. Render detectará automáticamente el archivo `render.yaml` de la raíz y mostrará los dos servicios a crear:
   - `tp-mesas-backend` (Web Service - Python)
   - `tp-mesas-frontend` (Static Site)
5. Hacer clic en **Apply**. Render compilará e iniciará ambos servicios automáticamente.

---

## Opción B — Despliegue Manual Servicio por Servicio

Si preferís crear los servicios manualmente desde el panel de Render, seguí estos pasos:

### 1. Crear el Backend (FastAPI)

1. En el dashboard de Render, hacer clic en **New +** → **Web Service**.
2. Conectar el repositorio de GitHub.
3. Completar los siguientes campos:
   - **Name:** `tp-mesas-backend` (o el nombre que elijas).
   - **Region:** `Ohio (US East)` (o la más cercana).
   - **Branch:** `Develop` (o `main`).
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:**  
     ```bash
     pip install -r requirements.txt && python -m app.ml.train --data data/pares_materias.csv --out models/modelo.joblib
     ```
   - **Start Command:**  
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** `Free`
4. Hacer clic en **Create Web Service**.
5. Una vez finalizado el build, copiar la URL pública generada (por ejemplo: `https://tp-mesas-backend.onrender.com`).

### 2. Crear el Frontend (React / Vite)

1. En el dashboard de Render, hacer clic en **New +** → **Static Site**.
2. Conectar el mismo repositorio de GitHub.
3. Completar los siguientes campos:
   - **Name:** `tp-mesas-frontend`.
   - **Branch:** `Develop` (o `main`).
   - **Root Directory:** `frontend`
   - **Build Command:**  
     ```bash
     npm install && npm run build
     ```
   - **Publish Directory:** `dist`
4. En la sección **Environment Variables**, agregar:
   - **Key:** `VITE_API_URL`
   - **Value:** La URL del backend obtenida en el paso anterior (ej. `https://tp-mesas-backend.onrender.com`).
5. En la sección **Redirects/Rewrites**, agregar una regla de tipo *Rewrite* para soportar SPA routing si fuera necesario:
   - **Source:** `/*`
   - **Destination:** `/index.html`
6. Hacer clic en **Create Static Site**.

---

## Consideraciones sobre el Plan Gratuito (Free Tier)

> [!NOTE]
> **Reposo por inactividad (Spin down / Cold start):**  
> Los servicios gratuitos de Render entran en modo reposo si pasan más de 15 minutos sin recibir tráfico. Cuando un usuario ingresa tras ese periodo, la primera petición tardará entre **40 y 50 segundos** en responder mientras se reactiva el contenedor del backend. Las siguientes peticiones responderán de forma instantánea.

> [!TIP]
> **CORS preconfigurado:**  
> El backend ya viene configurado para aceptar peticiones desde cualquier subdominio `https://*.onrender.com` y `localhost:5173`. Si desplegás el frontend en un dominio personalizado o en Vercel, podés definir la variable de entorno `CORS_ORIGINS=https://tu-dominio.com` en el servicio del backend en Render.
