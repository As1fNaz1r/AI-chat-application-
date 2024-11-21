from transformers import pipeline

def query_api(user_query, rag_system):
    relevant_context = rag_system.get_relevant_context(user_query)
    
    # Use Hugging Face's free API for text generation
    generator = pipeline('text-generation', model='gpt2')
    
    prompt = f"Context: {relevant_context}\n\nQuestion: {user_query}\n\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Extract the generated answer from the response
    answer = response.split("Answer:")[-1].strip()
    
    return answer