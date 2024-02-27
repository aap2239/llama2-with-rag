import chainlit as cl
import litellm

from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub

from langchain.vectorstores.chroma import Chroma

prompt = hub.pull("rlm/rag-prompt-llama")


# load the LLM
def load_llm():
    llm = Ollama(
        model="llama2",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(
    llm,
    vectorstore,
):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationSummaryMemory(llm=llm),  # Add memory for context
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True,
            "return_source_documents": True,
        },
    )
    return qa_chain


def qa_bot():
    llm = load_llm()
    DB_PATH = "vectorstores/db/"
    ollama_embeddings = OllamaEmbeddings(model="llama2")
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=ollama_embeddings
    )
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start_chat():
    chain = qa_bot()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"],
    )
    cb.answer_reached = True

    res = await chain.acall(message, callbacks=[cb])
    print(f"response: {res}")
    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()
