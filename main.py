from langchain_core.messages import HumanMessage
from src.RelevantDocuments import get_relevant_documents
import src.CacheMessages as CacheMessages
from src.ModelCall import app
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import logging
import uuid

retriever = get_relevant_documents()
logger = logging.getLogger(__name__)
flask_app = Flask(__name__)
cors = CORS(flask_app)
flask_app.config['CORS_HEADERS'] = 'Content-Type'


@flask_app.route("/<thread_id>", methods=['POST'])
@cross_origin()
def message(thread_id):
    request_id = str(uuid.uuid4())
    query = request.form['query']

    logger.info(f'Request from thread_id({thread_id}) uuid({request_id}) - Q: {query}')

    input_messages = CacheMessages.get_messages_from_user(thread_id) + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "question": query, "context": retriever.invoke(query)},
        {"configurable": {"thread_id": thread_id}},
    )

    CacheMessages.set_messages_from_user(thread_id, output["messages"])

    response_text = output["messages"][-1].content

    logger.info(f'Request from thread_id({thread_id}) uuid({request_id}) - R: {response_text}')
    return {
        'response': response_text
    }

def main():
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )
    logger.info('Started')

    flask_app.run(port=81, host='0.0.0.0')

    logger.info('Finishing')

if __name__ == '__main__':
    main()
