import fitz  # PyMuPDF
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langdetect import detect
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
import os
import httpx
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

class GroqLLM(LLM):
    model_name: str = "llama3-70b-8192"
    temperature: float = 0.7
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        with httpx.Client() as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

def preprocess_text(text, lang):
    lines = text.split('\n')
    processed_lines = [line for line in lines if not line.strip().startswith(('Header', 'Footer', 'Page'))]
    processed_text = ' '.join(processed_lines)
    
    if lang == 'ar':
        reshaped_text = arabic_reshaper.reshape(processed_text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    return processed_text

async def process_pdf(file):
    doc = fitz.open(file.path)
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text()
    
    lang = detect(pdf_text)
    preprocessed_text = preprocess_text(pdf_text, lang)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    file_texts = text_splitter.split_text(preprocessed_text)
    
    file_metadatas = [{"source": f"{i}-{file.name}", "lang": lang} for i in range(len(file_texts))]
    
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
    
    msg = cl.Message(content=f"Processing {len(files)} files... This may take a while for large or complex PDFs.")
    await msg.send()

    results = await asyncio.gather(*[process_pdf(file) for file in files])
    texts = []
    metadatas = []
    for file_texts, file_metadatas in results:
        texts.extend(file_texts)
        metadatas.extend(file_metadatas)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        GroqLLM(),
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
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]
            
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
        await cl.Message(content=answer, elements=text_elements).send()
    except asyncio.TimeoutError:
        await cl.Message(content="I'm sorry, but the request timed out. The PDF might be too large or complex to process quickly. Could you try asking about a specific part of the document?").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}. Please try again or contact support if the problem persists.").send()