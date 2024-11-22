from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente de resposta a perguntas. Use os seguintes pedaços de contexto recuperado para responder à pergunta. Se você não sabe a resposta, apenas diga que não sabe. Seja conciso.\nPergunta: {question} \nContexto: {context} \nResposta:",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
