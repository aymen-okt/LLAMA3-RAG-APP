import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from langdetect import detect
import arabic_reshaper
from bidi.algorithm import get_display
from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader

def preprocess_text(text, lang):
    lines = text.split('\n')
    processed_lines = [line for line in lines if not line.strip().startswith(('Header', 'Footer', 'Page'))]
    processed_text = ' '.join(processed_lines)
    
    if lang == 'ar':
        reshaped_text = arabic_reshaper.reshape(processed_text)
        return get_display(reshaped_text)
    return processed_text

def split_and_process_text(text, lang):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    if lang == 'ar':
        return [get_display(arabic_reshaper.reshape(chunk)) for chunk in chunks]
    return chunks

async def process_file(file):
    loader = UnstructuredFileLoader(file.path)
    documents = loader.load()
    
    file_text = " ".join([doc.page_content for doc in documents])
    
    lang = detect(file_text)
    preprocessed_text = preprocess_text(file_text, lang)
    file_texts = split_and_process_text(preprocessed_text, lang)
    
    file_metadatas = [{"source": f"{i}-{file.name}", "lang": lang} for i in range(len(file_texts))]
    
    return file_texts, file_metadatas

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more files to begin!",
            accept=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
                "text/plain",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
                "text/csv",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint"
            ],
            max_size_mb=500,
            max_files=10,
            timeout=180,
        ).send()
    
    msg = cl.Message(content=f"Processing {len(files)} files... This may take a while for large or complex files.")
    await msg.send()

    results = await asyncio.gather(*[process_file(file) for file in files])
    texts = []
    metadatas = []
    for file_texts, file_metadatas in results:
        texts.extend(file_texts)
        metadatas.extend(file_metadatas)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="qwen2:7b"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    await cl.Message(content="Processing done. You can now ask questions!").send()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    
    input_language = detect(message.content)
    prompt = f"Respond in {input_language}. User question: {message.content}"
    
    try:
        res = await asyncio.wait_for(chain.ainvoke(prompt, callbacks=[cb]), timeout=60)
        answer = res["answer"]
        source_documents = res["source_documents"]
        text_elements = []
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
            source_names = [text_el.name for text_el in text_elements]
            answer += f"\nSources: {', '.join(source_names)}" if source_names else "\nNo sources found"
        await cl.Message(content=answer, elements=text_elements).send()
    except asyncio.TimeoutError:
        await cl.Message(content="I'm sorry, but the request timed out. The file might be too large or complex to process quickly. Could you try asking about a specific part of the document?").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}. Please try again or contact support if the problem persists.").send()