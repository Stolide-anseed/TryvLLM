from Rag.vector_store import create_collection,upsert_chunks
from Rag.embedder import Embedder

'''
docker run -d --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  qdrant/qdrant
'''

model = Embedder(model_name='intfloat/multilingual-e5-small')


text = 'Приёмный отец По. Он рассказывает, как нашёл По младенцем в ящике с редисом. Его любовь показывает, что семья определяется не только происхождением.'
vector = model.encode_query('Приёмный отец По. Он рассказывает, как нашёл По младенцем в ящике с редисом. Его любовь показывает, что семья определяется не только происхождением.')


collection_name = 'demo_qudrant'

# create_collection(vector_size=384, collection_name='demo_qudrant')

upsert_chunks(
    chunks=[{
        'chunk_id': 'Panda_plot01',
        'title':'Kungu-fu Panda',
        'text': text
    }],
    vectors=[vector],
    collection_name=collection_name
)