# Modules to support email application
import email
from email.parser import BytesParser
from bs4 import BeautifulSoup
import docx
import json
import pdfplumber
import io
import re
import imaplib
import datetime

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.

def store_email_file_in_vectorstore(texts,url=os.environ["NEO4J_URI"],username=os.environ["NEO4J_USERNAME"],password=os.environ["NEO4J_PASSWORD"],destroy_information_flag=False):
        vectorstore = Neo4jVector.from_texts(
                texts=texts, 
                embedding=embeddings, 
                url=url, 
                username=username, 
                password=password, 
                index_name="SRRP_email", 
                node_label="SRRP_email",
                pre_delete_collection=destroy_information_flag  # Delete existing data? True or False
        )
        return(vectorstore)

# Support IMAP application
imap_server = os.getenv("IMAP_SERVER")
email_account = os.getenv("EMAIL_ACCOUNT")
imap_password = os.getenv("IMAP_PASSWORD")
address_book = json.loads(os.getenv("ADDRESS_BOOK"))
days_to_search = int(os.getenv("DAYS_TO_SEARCH"))

def clean_text(text):
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    text = re.sub(non_ascii_pattern, '', text).replace('\t', ' ').replace('\n', ' ').replace('\\t', ' ').replace('\\n', ' ')
    text = re.sub(r'\s+',' ',text)
    return(text)

def docx2text(docx_bytes):
    """
    Extracts and returns the text multipart_content from a DOCX file.

    :param docx_bytes: Bytes content of a DOCX file.
    :return: Plain text extracted from the DOCX file.
    """
    doc = docx.Document(io.BytesIO(docx_bytes))
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return clean_text('\n'.join(full_text))

def pdf2text(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return clean_text(text)

def html2text(html_content):
    """
    Extracts and returns the text content from HTML.

    :param html_content: String containing HTML content.
    :return: Plain text extracted from HTML.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    return clean_text(soup.get_text())

def filter_email_parts(eml_file_path):
    """
    Reads an EML file, filters out parts not matching specified MIME types.

    :param eml_file_path: Path to the .eml file
    :return: A list of text strings that represent the text contents of the email file
    """
    with open(eml_file_path, 'rb') as f:
        msg = BytesParser().parse(f)

    content = []
    multipart_content = []

    if msg.is_multipart():
        for part in msg.walk():
            mime_type=part.get_content_type()
            if mime_type == "application/pdf":
                pdf_bytes = part.get_payload(decode=True)
                pdf_text = pdf2text(pdf_bytes)
                multipart_content.append(pdf_text)
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                docx_bytes = part.get_payload(decode=True)
                word_text = docx2text(docx_bytes)
                multipart_content.append(word_text)
            elif mime_type == "text/html":
                html_content = part.get_payload(decode=True).decode('utf-8')
                text_content = html2text(html_content)
                content.append(text_content)
            elif mime_type == "text/plain":
                multipart_content.append(clean_text(part.get_payload(decode=True).decode('utf-8')))
    else:
        mime_type=msg.get_content_type()
        if mime_type == "text/html":
            html_content = msg.get_payload(decode=True).decode('utf-8')
            text_content = html2text(html_content)
            content.append(text_content)
        elif mime_type == "text/plain":
            content.append(clean_text(msg.get_payload(decode=True).decode('utf-8')))

    content = multipart_content if multipart_content != [] else content
    return(' '.join(content))

def process_email_file(email_filename):
    texts = filter_email_parts(email_filename)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_text(texts)
    return(docs)

def process_email(imap_server, email_account, password, from_addresses, days):

    tmp_filepath="/tmp/SRRP_email"
    destroy_information_flag = True

    # Connect to IMAP server
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_account, password)
    mail.select('inbox')

    # Calculate the date X days ago
    current_date = datetime.datetime.now()
    date_days_ago = (current_date - datetime.timedelta(days=days)).strftime("%d-%b-%Y")
    current_date = current_date.strftime("%d-%b-%Y")
    logger.info(
        f"Searching for email between {date_days_ago} and {current_date}\n",
    )

    mail_ids = []
    for address in from_addresses:
        # Search emails from a specific address since the date
        _, data = mail.search(None, f'(FROM "{address}" SINCE "{date_days_ago}")')
        l = data[0].split()
        n = len(l)
        n_str = f"{n} number of" if (n>0) else "no"
        logger.info(
            f"{address} sent {n_str} emails...",
        )
        mail_ids.extend(l)

    total_emails = len(mail_ids)
    logger.info(f"\nTotal emails to process: {total_emails}\nNow vectorizing...\n")

    # Process each email
    for i, num in enumerate(mail_ids, 1):  # Start counting from 1
        _, data = mail.fetch(num, '(RFC822)')
        raw_email = data[0][1]

        # Write each email to a separate file
        email_filename = f"{tmp_filepath}_{num.decode('utf-8')}.eml"
        with open(email_filename, 'wb') as email_file:
            email_file.write(raw_email)

        docs = process_email_file(email_filename)
        os.remove(email_filename)
        vectorstore = store_email_file_in_vectorstore(docs,destroy_information_flag=destroy_information_flag)
        destroy_information_flag = False if destroy_information_flag == True else destroy_information_flag        
        # Print the percent complete
        percent_complete = (i / total_emails) * 100
        if i % 10 == 0 or i == total_emails:
            logger.info(
                f"Processed {i}/{total_emails} emails ({percent_complete:.2f}% complete)",
            )

    # Close connections
    mail.logout()

    # Return reference to the vectorstore
    return(vectorstore)