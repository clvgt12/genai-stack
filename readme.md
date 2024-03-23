![Source: Dall-E 3](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhzRmIkIXnhveCCPdZ1b1SJ1d0mfj0rZsb6YtfVf04HKBjHvNDAxzoDJKzDAxmx75XwApXWKyAfhFm9YvPI1wIJAH5XVvOvMYYkztmVv-Z04XZfClofv3024UPj3cotJs_snjI51tKy5_mF2KQFlCfFTvITPDUFU8OUahX-GoCHBXz7zizMXV7K-Vnvoofz/s1600/tech-writer-header-image-dalle3.jpg "Image of a AI robot, typing at a computer, presumably writing a blog post article, and wanting an application to help it write more clearly, concisely and effectively")
# GenAI Stack fork

This is my fork of the GenAI stack, an open source [collection](https://github.com/docker/genai-stack) of tools that will jump start building GenAI applications.

## Ask Your DOCX

The stack offers several demo applications, one of which is 'pdf_bot'.  This implements a straightfoward "ask your document" use case.  It illustrates how to build a [RAG](https://www.k2view.com/blog/rag-genai/) application using the [neo4j](https://neo4j.com/) vector store, open source LLM and the [LangChain](https://github.com/langchain-ai/langchain) [Python](https://www.python.org/) framework.

I used pdf_bot as a baseline to create ask_your_docx, which extends the app to include parsing Microsoft DOCX and LibreOffice ODT files.  I also enhanced the REST API module with endpoints, then created a CLI program to run batch queries against documents.

Feel free to use this software if you have an interest.