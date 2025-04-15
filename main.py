import os
import sys
from symbol import file_input

import openai
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

sys.path.append('../..')


def load_env():
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']


def load_pdf():
    loader = PyPDFLoader("docs/machinelearning-lecture01.pdf")
    pages = loader.load()
    len(pages)
    page = pages[0]
    print(page.page_content[0:500])
    page.metadata


def load_youtube():
    url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir = "docs/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),  # fetch from youtube
        # FileSystemBlobLoader(save_dir, glob="*.m4a"),  # fetch locally
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print(docs)


def load_url():
    loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
    docs = loader.load()
    print(docs[0].page_content[:500])


def text_splitter():
    chunk_size = 26
    chunk_overlap = 4

    c_splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator=' '
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    text1 = 'abcdefghijklmnopqrstuvwxyz'
    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    # print(r_splitter.split_text(text3))
    some_text = """When writing documents, writers will use document structure to group content. \
    This can convey to the reader, which idea's are related. For example, closely related ideas \
    are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
    Paragraphs are often delimited with a carriage return or two carriage returns. \
    Carriage returns are the "backslash n" you see embedded in this string. \
    Sentences have a period at the end, but also, have a space.\
    and words are separated by space."""

    print(len(some_text))
    c_splitter.split_text(some_text)
    r_splitter.split_text(some_text)


def pdf_splitter():
    loader = PyPDFLoader("docs/MachineLearning-Lecture01.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    print(len(docs))
    print(len(pages))


def loading_data():
    # Load PDF
    loaders = [
        PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("docs/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("docs/MachineLearning-Lecture02.pdf"),
        PyPDFLoader("docs/MachineLearning-Lecture03.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    splits = text_splitter.split_documents(docs)


def embedding_simulation():
    from langchain_community.embeddings import OpenAIEmbeddings
    embedding = OpenAIEmbeddings()

    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"

    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)

    import numpy as np

    np.dot(embedding1, embedding2)
    np.dot(embedding1, embedding3)
    np.dot(embedding2, embedding3)


# def store_in_vector_stores(embedding):
#     from langchain.vectorstores import Chroma
#     persist_directory = 'docs/chroma/'
#     # remove old database files if any
#     # !rm - rf. / docs / chroma
#
#     vectordb = Chroma.from_documents(
#         documents=splits,
#         embedding=embedding,
#         persist_directory=persist_directory
#     )
#
#     print(vectordb._collection.count())


def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


import panel as pn
import param


class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.loaded_file = "docs/cs229_lectures/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer']
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  # clears loading indicator when cleared
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist = [pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, count=0):
        self.chat_history = []
        return



if __name__ == '__main__':
    load_env()
    cb = cbfs()

    file_input = pn.widgets.FileInput(accept='.pdf')
    button_load = pn.widgets.Button(name="Load DB", button_type='primary')
    button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
    button_clearhistory.on_click(cb.clr_history)
    inp = pn.widgets.TextInput(placeholder='Enter text here…')

    bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
    conversation = pn.bind(cb.convchain, inp)

    jpg_pane = pn.pane.Image('./img/convchain.jpg')

    tab1 = pn.Column(
        pn.Row(inp),
        pn.layout.Divider(),
        pn.panel(conversation, loading_indicator=True, height=300),
        pn.layout.Divider(),
    )
    tab2 = pn.Column(
        pn.panel(cb.get_lquest),
        pn.layout.Divider(),
        pn.panel(cb.get_sources),
    )
    tab3 = pn.Column(
        pn.panel(cb.get_chats),
        pn.layout.Divider(),
    )
    tab4 = pn.Column(
        pn.Row(file_input, button_load, bound_button_load),
        pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
        pn.layout.Divider(),
        pn.Row(jpg_pane.clone(width=400))
    )
    dashboard = pn.Column(
        pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
        pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4))
    )
    dashboard
