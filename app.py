from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7,

)
response = llm.invoke("What is RAG?") #invoke gives AI message, & returns a coplete class
print(response.content) 