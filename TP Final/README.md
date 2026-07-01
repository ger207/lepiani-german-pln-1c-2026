\---

title: Asistente Constitucion RAG

emoji: ⚖️

colorFrom: blue

colorTo: indigo

sdk: gradio

sdk\_version: 5.33.0

app\_file: app.py

pinned: false

\---





Asistente de la Constitución de la Nación Argentina mediante RAG

Integrante

Germán Pablo Le Piani

Materia

Laboratorio de PLN: Analítica, Textos y Cultura



Año

2026



Contexto del proyecto

Los ciudadanos que desean conocer las leyes fundamentales de nuestro país y sus derechos suelen encontrarse con un documento extenso y complejo de navegar.



Para facilitar este proceso, se desarrolló un asistente basado en la técnica Retrieval-Augmented Generation (RAG), capaz de responder preguntas utilizando exclusivamente la información contenida en documentos especializados\[cite: 2].



Objetivo

Desarrollar un asistente educativo que permita consultar conceptos mediante lenguaje natural, reduciendo la necesidad de leer extensos documentos para encontrar definiciones o explicaciones específicas\[cite: 2].



¿Para qué sirve?

El sistema permite que cualquier usuario pueda realizar preguntas y obtener respuestas rápidas basadas en la documentación cargada\[cite: 2]. En este caso, facilita la consulta sobre cómo se dictó nuestra Constitución, qué derechos nos amparan y cómo se organiza el Estado.



Algunos ejemplos de consultas son:



¿Qué dice el documento sobre los derechos de los trabajadores?

¿Cómo se formó o se dictó la Constitución?

¿Cómo se compone el Poder Legislativo?

¿Qué es el recurso de amparo?

¿Con qué documentos trabaja?

El sistema utiliza el documento PDF de la Constitución de la Nación Argentina. Este documento es cargado e indexado para permitir búsquedas semánticas y recuperación de información relevante.



Funcionamiento general

El sistema sigue las siguientes etapas:



Carga de documentos PDF.

Extracción del contenido textual.

División del texto en fragmentos (chunks).

Generación de embeddings.

Almacenamiento en ChromaDB.

Recuperación de los fragmentos más relevantes ante una consulta.

Generación de una respuesta utilizando un modelo de lenguaje.

Presentación de la respuesta junto con las fuentes consultadas.

Interfaz de usuario

El sistema cuenta con una interfaz desarrollada en Gradio, accesible desde navegador web.



A través de esta interfaz el usuario puede:



Cargar uno o varios documentos PDF.

Indexar los documentos para su procesamiento.

Realizar preguntas en lenguaje natural.

Visualizar la respuesta generada por el sistema.

Consultar las fuentes utilizadas para elaborar la respuesta.

La interfaz fue desplegada en Hugging Face Spaces, permitiendo acceder al asistente sin necesidad de instalar software adicional.



Tecnologías utilizadas

Python

LangChain

ChromaDB

Sentence Transformers

Ollama (desarrollo y pruebas locales)

Hugging Face Inference API (despliegue en Spaces)

Gradio

PyPDF

Limitaciones observadas

Durante las pruebas se observó que el sistema funciona mejor cuando las consultas son específicas y están relacionadas con los conceptos presentes en los documentos.



Observamos que el sistema puede tener dificultades cuando:



La información está distribuida en varios fragmentos del documento.

Se utilizan términos muy diferentes a los presentes en los PDF.

El texto del PDF no puede extraerse correctamente.

La respuesta existe en el documento pero no es recuperada entre los fragmentos más relevantes.

Asimismo, una respuesta incorrecta no necesariamente implica que el modelo desconozca el tema, sino que puede deberse a que no se recuperaron los fragmentos adecuados para responder la consulta. Por este motivo, las respuestas deben interpretarse como una ayuda para la consulta y no como un reemplazo del documento original.



Conclusiones

Logramos desarrollar un asistente RAG capaz de responder preguntas utilizando exclusivamente la información contenida en documentos PDF. Las pruebas realizadas mostraron buenos resultados para preguntas directas y definiciones específicas, permitiendo acceder rápidamente a información que normalmente se encuentra dispersa.



El proyecto permitió aplicar conceptos de procesamiento de lenguaje natural, embeddings, recuperación semántica, bases vectoriales, modelos de lenguaje e interfaces conversacionales.



Despliegue

El sistema fue publicado en Hugging Face Spaces y puede utilizarse desde un navegador web sin necesidad de instalar software adicional.



URL: (https://huggingface.co/spaces/ger93ifts/ger\_ifts\_gradio)

