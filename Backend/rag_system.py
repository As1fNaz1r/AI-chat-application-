# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# class RAGSystem:
#     def __init__(self, knowledge_base):
#         self.knowledge_base = knowledge_base
#         self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#         self.generator = pipeline('text-generation', model='gpt2')

#     def get_relevant_context(self, query, top_k=3):
#         query_embedding = self.model.encode(query)
        
#         similarities = [np.dot(query_embedding, doc["embedding"]) for doc in self.knowledge_base]
#         top_indices = np.argsort(similarities)[-top_k:]
        
#         relevant_context = [self.knowledge_base[i]["text"] for i in top_indices]
#         return " ".join(relevant_context)

#     def get_answer(self, query):
#         relevant_context = self.get_relevant_context(query)
#         prompt = f"Context: {relevant_context}\n\nQuestion: {query}\n\nAnswer:"
#         response = self.generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
#         answer = response.split("Answer:")[-1].strip()
#         return answer

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGSystem:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')  # Force CPU usage
        self.generator = pipeline('text-generation', model='gpt2', device=-1)  # Force CPU usage

    def get_relevant_context(self, query, top_k=3):
        query_embedding = self.model.encode(query, device='cpu')  # Force CPU usage

        similarities = [np.dot(query_embedding, doc["embedding"]) for doc in self.knowledge_base]
        top_indices = np.argsort(similarities)[-top_k:]

        relevant_context = [self.knowledge_base[i]["text"] for i in top_indices]
        return " ".join(relevant_context)

    def get_answer(self, query):
        relevant_context = self.get_relevant_context(query)
        prompt = f"Context: {relevant_context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        answer = response.split("Answer:")[-1].strip()
        return answer
