from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="minimax-m2:cloud",
    temperature=0.7,
    num_ctx= 5200   
    
)

prompt = ChatPromptTemplate.from_messages([
    
    ("system", "You are a helpful assistant and an expert in AI. You will answer the user's question in a concise and informative manner."),
    ("human", "{question}")
])
chain = prompt | llm  | StrOutputParser()
 # create a chain that combines the prompt and the language model LCEL


# response = chain.invoke({"question": "What is NLP?"})
# print(response)
for chunk in chain.stream({"question": "What is NLP?"}):
    print(chunk, end="", flush=True)