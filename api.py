from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import json
import numpy as np
import pandas as pd

app = FastAPI()

# Permitir CORS para Astro
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y datos KNN (MATCH)
with open('data/modelo_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('data/matriz_normalizada.pkl', 'rb') as f:
    matriz_normalizada = pickle.load(f)

with open('data/matriz_final.pkl', 'rb') as f:
    matriz_final = pickle.load(f)

with open('data/datos_recomendador.json', 'r') as f:
    datos = json.load(f)

usuarios = datos['usuarios']
top_cursos = datos['cursos']
nombres_cursos = datos['nombres_cursos']

usuario_a_indice = {usuario: idx for idx, usuario in enumerate(usuarios)}

# Cargar modelos TF-IDF (DIVING)
with open('data/tfidf_model.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('data/similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# Cargar DataFrame de cursos
df_cursos_diving = pd.read_json('data/df_cursos.json')


# ENDPOINT 1: Recomendaciones por usuario (MATCH)
@app.post("/recomendar")
async def recomendar(usuario: dict):
    nombre_usuario = usuario.get("nombre")
    
    if nombre_usuario not in usuario_a_indice:
        return {"error": f"Usuario '{nombre_usuario}' no encontrado"}
    
    target_user_index = usuario_a_indice[nombre_usuario]
    distances, indices = knn.kneighbors(
        matriz_normalizada[target_user_index].reshape(1, -1), 
        n_neighbors=3
    )
    
    neighbors_ratings = matriz_final[indices.flatten()]
    predicted_ratings = np.zeros(neighbors_ratings.shape[1])
    
    for i in range(neighbors_ratings.shape[1]):
        non_zero = neighbors_ratings[:, i][neighbors_ratings[:, i] != 0]
        if len(non_zero) > 0:
            predicted_ratings[i] = non_zero.mean()
    
    unrated_items = np.where(matriz_final[target_user_index] == 0)[0]
    predicted_unrated = predicted_ratings[unrated_items]
    sorted_indices = np.argsort(predicted_unrated)[::-1]
    top_5_indices = sorted_indices[:5]
    top_5_items = unrated_items[top_5_indices]
    
    recomendaciones = []
    for rank, item_idx in enumerate(top_5_items, 1):
        course_id = top_cursos[item_idx]
        course_name = nombres_cursos.get(course_id, "Desconocido")
        rating = float(predicted_ratings[item_idx])
        
        recomendaciones.append({
            "rank": rank,
            "nombre": course_name,
            "calificacion": round(rating, 2)
        })
    
    return {"usuario": nombre_usuario, "recomendaciones": recomendaciones}


# ENDPOINT 2: Cursos similares por sigla (DIVING)
@app.post("/similares")
async def similares(curso: dict):
    sigla = curso.get("sigla").upper()
    n_recomendaciones = 10
    
    try:
        # Buscar el índice del curso por sigla
        idx = df_cursos_diving[df_cursos_diving['sigle'].str.upper() == sigla].index[0]
        
        # Obtener similitudes del curso con todos los demás
        similitudes = similarity_matrix[idx]
        
        # Ordenar de mayor a menor similitud (excluir el mismo curso)
        indices_similares = np.argsort(similitudes)[::-1][1:n_recomendaciones+1]
        
        # Crear respuesta
        similares = []
        for rank, idx_similar in enumerate(indices_similares, 1):
            curso_similar = df_cursos_diving.iloc[idx_similar]
            similitud_pct = int(similitudes[idx_similar] * 100)
            
            similares.append({
                "rank": rank,
                "sigle": curso_similar['sigle'].upper(),
                "nombre": curso_similar['name'],
                "area": curso_similar['area'],
                "similitud": similitud_pct
            })
        
        return {"sigla": sigla, "similares": similares}
        
    except IndexError:
        return {"error": f"Curso '{sigla}' no encontrado"}
    except Exception as e:
        return {"error": str(e)}

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)