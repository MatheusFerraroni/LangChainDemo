import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def get_relevant_documents():
    loader = WebBaseLoader(
        web_paths=(
            'https://oficial.unimar.br/cursos/administracao/',
            'https://oficial.unimar.br/cursos/analise-e-desenvolvimento-de-sistemas/',
            'https://oficial.unimar.br/cursos/arquitetura-e-urbanismo/',
            'https://oficial.unimar.br/cursos/biomedicina/',
            'https://oficial.unimar.br/cursos/ciencia-da-computacao/',
            'https://oficial.unimar.br/cursos/ciencias-contabeis/',
            'https://oficial.unimar.br/cursos/design-grafico/',
            'https://oficial.unimar.br/cursos/direito/',
            'https://oficial.unimar.br/cursos/educacao-fisica/',
            'https://oficial.unimar.br/cursos/enfermagem/',
            'https://oficial.unimar.br/cursos/engenharia-agronomica/',
            'https://oficial.unimar.br/cursos/engenharia-civil/',
            'https://oficial.unimar.br/cursos/engenharia-de-producao-mecanica/',
            'https://oficial.unimar.br/cursos/engenharia-eletrica/',
            'https://oficial.unimar.br/cursos/farmacia/',
            'https://oficial.unimar.br/cursos/fisioterapia/',
            'https://oficial.unimar.br/cursos/bacharelado-em-inteligencia-artificial/',
            'https://oficial.unimar.br/cursos/medicina/',
            'https://oficial.unimar.br/cursos/medicina-veterinaria/',
            'https://oficial.unimar.br/cursos/nutricao/',
            'https://oficial.unimar.br/cursos/odontologia/',
            'https://oficial.unimar.br/cursos/psicologia/',
            'https://oficial.unimar.br/cursos/publicidade-e-propaganda/',
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=('sobre', 'ficha', 'duvidas', 'modal-body')
            )
        ),
        requests_kwargs={"verify": False}
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever
