import PyPDF2
import asyncio
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

def preprocess_text(text):
    # Remove headers, footers, and page numbers
    lines = text.split('\n')
    processed_lines = [line for line in lines if not line.strip().startswith(('Header', 'Footer', 'Page'))]
    return ' '.join(processed_lines)

async def process_pdf(file):
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    preprocessed_text = preprocess_text(pdf_text)
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    file_texts = text_splitter.split_text(preprocessed_text)
    
    # Create metadata for each chunk
    file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
    
    return file_texts, file_metadatas

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=500,
            max_files=10,
            timeout=180,
        ).send()

    results = await asyncio.gather(*[process_pdf(file) for file in files])

    texts = []
    metadatas = []
    for file_texts, file_metadatas in results:
        texts.extend(file_texts)
        metadatas.extend(file_metadatas)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama3"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    elements = [cl.Image(name="image", display="inline", path="pic.png")]
    msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!", elements=elements)
    await msg.send()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
