from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

SYSTEM_PROMPT = """You are a research assistant. Answer the user's question 
using ONLY the provided context from research papers. If the context doesn't 
contain enough information to answer, say so honestly.

For each claim you make, reference which source it came from. Keep your 
answers clear and technically precise.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

chain = prompt | llm


def answer_question(question: str, retrieved: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    given a question and retrieved chunks, generate an answer
    and return it along with the source chunks used.
    """
    # build context string from retrieved chunks
    context_parts = []
    for i, doc in enumerate(retrieved):
        source_label = f"[{doc['source']}, p.{doc.get('page', '?')}]"
        context_parts.append(f"Source {i+1} {source_label}:\n{doc['text']}")

    context = "\n\n".join(context_parts)

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    # return the sources that were actually used
    sources = [
        {
            "text": doc["text"][:300],  # truncate for response size
            "source": doc["source"],
            "page": doc.get("page"),
        }
        for doc in retrieved
    ]

    return response.content, sources
