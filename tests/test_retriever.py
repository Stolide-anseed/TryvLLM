from Rag.retriever import Retriever


retriever = Retriever(model_name='intfloat/multilingual-e5-small',collection_name='movies')


print(retriever.retieve(query="Почему Шрэку в Китае не поставили неистовую пятёрку?",top_k=5))