import os
import sys
from pathlib import Path

# Ensure project root is on path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval.hybrid_search import _load_bm25, qdrant_search, bm25_search
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.rag_generate import generate_answer

def run_tests():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Define OPENROUTER_API_KEY.")
        return

    qdrant_url = "http://localhost:6333"
    collection = "catalog_es"
    bm25_path = Path("data/index/bm25.pkl")
    model_name = "intfloat/multilingual-e5-small"

    print("Loading indices...")
    bm25, bm25_chunks = _load_bm25(bm25_path)
    client = QdrantClient(url=qdrant_url)
    model = SentenceTransformer(model_name)

    test_queries = [
        # Prueba 1: Skinnia (Página 4) - Test de extracción de tablas (precio y tallas)
        "¿Cuáles son los tamaños disponibles y los precios para el apósito protector en spray Skinnia (SKNPO1/SKNPO2)?",
        
        # Prueba 2: Pet Head (Página 9/11) - Test de descripciones y SKUs
        "¿Qué características tiene el champú Furtastic y cuál es su número de referencia?",
        
        # Prueba 3: Coachi (Página 20) - Test de contexto general de la marca
        "¿Quién diseñó la gama Coachi de adiestramiento? Explica para qué sirve el silbato profesional.",
        
        # Prueba 4: Andis (Página 40) - Test de detalles técnicos en tablas
        "Dame el peso, los vatios y el precio de la máquina Corta Pelos Andis AGC2 negra (referencia CU03022).",
        
        # Prueba 5: Múltiples productos / Comparación
        "Enumera todos los recambios disponibles para la máquina Andis DBLC-2 Pulse ZR II y sus precios correspondientes."
    ]

    for i, query in enumerate(test_queries, 1):
        print("\n" + "*"*80)
        print(f"TEST {i}: {query}")
        print("*"*80)
        
        # Buscar
        vec = qdrant_search(client, collection, model, query, top_k=8)
        kw = bm25_search(bm25, bm25_chunks, query, top_k=8)
        fused = reciprocal_rank_fusion(vec, kw, top_n=4)
        
        retrieved_chunks = [item.payload for item in fused]
        
        # Imprimir qué chunks fueron encontrados
        print("\n[Retrieved Documents]:")
        for chunk in retrieved_chunks:
            print(f"- Page {chunk.get('page_number', '?')} (Ref: {chunk.get('source_file')})")
        
        # Generar respuesta
        generate_answer(query, retrieved_chunks, api_key)

if __name__ == "__main__":
    run_tests()
