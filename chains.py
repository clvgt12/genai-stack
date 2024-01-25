from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
    BedrockEmbeddings,
)
from langchain.chat_models import ChatOpenAI, ChatOllama, BedrockChat
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from typing import List, Any
from utils import BaseLogger, extract_title_and_question

from langchain import hub

def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{query}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"query": user_input}, config={"callbacks": callbacks}
        ).content
        return {"result": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
    #   System: Always talk in pirate speech.
    # general_system_template = """ 
    # Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # ----
    # {summaries}
    # ----
    # """
    # general_user_template = "Question:```{question}```"
    # messages = [
    #     SystemMessagePromptTemplate.from_template(general_system_template),
    #     HumanMessagePromptTemplate.from_template(general_user_template),
    # ]
    # qa_prompt = ChatPromptTemplate.from_messages(messages)

    # qa_chain = load_qa_with_sources_chain(
    #     llm,
    #     chain_type="stuff",
    #     prompt=qa_prompt,
    # )
    prompt = hub.pull("rlm/rag-prompt")

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="pdf_bot"  # vector by default
    #     index_name="stackoverflow",  # vector by default
    #     text_node_property="body",  # text by default
    #     retrieval_query="""
    # WITH node AS question, score AS similarity
    # CALL  { with question
    #     MATCH (question)<-[:ANSWERS]-(answer)
    #     WITH answer
    #     ORDER BY answer.is_accepted DESC, answer.score DESC
    #     WITH collect(answer)[..2] as answers
    #     RETURN reduce(str='', answer IN answers | str + 
    #             '\n### Answer (Accepted: '+ answer.is_accepted +
    #             ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    # } 
    # RETURN '##Question: ' + question.title + '\n' + question.body + '\n' 
    #     + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    # ORDER BY similarity ASC // so that best answers are the last
    # """,
    )

    # kg_qa = RetrievalQAWithSourcesChain(
    #     combine_documents_chain=qa_chain,
    #     # retriever=kg.as_retriever(search_kwargs={"k": 2}),
    #     retriever=kg.as_retriever(),
    #     reduce_k_below_max_tokens=False,
    #     max_tokens_limit=3375,
    # )
    kg_qa = RetrievalQA.from_chain_type(
        llm,
        retriever=kg.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return kg_qa


def generate_ticket(neo4j_graph, llm_chain, input_question):
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. \n{question[0]}\n----\n\n"
        questions_prompt += f"{question[1][:150]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Formulate a question in the same style and tone as the following example questions.
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return format template:
    ---
    Title: This is a new title
    Question: This is a new question
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{input_question}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def configure_email_qa_rag_chain(model, vectorstore):
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | model
        | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | model
        | StrOutputParser(),
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }
# And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return(final_chain)

def save_memory(question, result):
    memory.save_context({"question": question}, {"answer": result["answer"].content})
    memory.load_memory_variables({})

def ask_question(question, final_chain):
    inputs = {"question": question}
    result = final_chain.invoke(inputs)
    save_memory(question, result)
    return(result)