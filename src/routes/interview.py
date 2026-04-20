from fastapi import APIRouter, UploadFile, Request, status
from fastapi.responses import JSONResponse
from uuid import UUID
import fitz  # PyMuPDF
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.embeddings import CohereEmbeddings
from langchain_openai import ChatOpenAI
from helpers.config import get_settings
from pydantic import BaseModel
from typing import List

logger = logging.getLogger('uvicorn.error')

interview_router = APIRouter(
    prefix="/api/v1/interview",
    tags=["api_v1", "interview"],
)

@interview_router.post("/start/{project_id}")
async def start_interview(request: Request, project_id: UUID, file: UploadFile):
    try:
        # Step 1: Read PDF from memory
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        # Step 2: Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([text])

        # Step 3: Store in PGVector
        settings = get_settings()
        connection_string = settings.POSTGRES_URL

        embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-multilingual-v3.0"
        )

        collection_name = str(project_id).replace("-", "_")

        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=connection_string,
            pre_delete_collection=True,
        )

        # Step 4: Generate 5 questions
        llm = ChatOpenAI(
            model=settings.GENERATION_MODEL_ID,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_URL,
        )

        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        result = qa.run(
            "Based on this CV, generate exactly 5 technical interview questions. "
            "Return ONLY a JSON array of 5 strings, no extra text. Example: [\"Q1\", \"Q2\", \"Q3\", \"Q4\", \"Q5\"]"
        )

        questions = json.loads(result.strip())

        return JSONResponse(content={
            "signal": "interview_started",
            "project_id": str(project_id),
            "questions": questions
        })

    except Exception as e:
        logger.error(f"Interview start error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "interview_start_error", "error": str(e)}
        )
    
class EvaluationRequest(BaseModel):
    questions: List[str]
    answers: List[str]

@interview_router.post("/evaluate/{project_id}")
async def evaluate_interview(request: Request, project_id: UUID, eval_request: EvaluationRequest):
    try:
        settings = get_settings()
        
        # Step 1: Load the existing PGVector collection
        embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-multilingual-v3.0"
        )
        
        collection_name = str(project_id).replace("-", "_")
        
        vectorstore = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=settings.POSTGRES_URL,
        )
        
        # Step 2: Build Q&A pairs string
        qa_pairs = "\n".join([
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(eval_request.questions, eval_request.answers))
        ])
        
        # Step 3: Retrieve relevant CV context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        cv_docs = retriever.get_relevant_documents(qa_pairs)
        cv_context = "\n".join([doc.page_content for doc in cv_docs])
        
        # Step 4: Generate evaluation
        llm = ChatOpenAI(
            model=settings.GENERATION_MODEL_ID,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_URL,
        )
        
        prompt = f"""You are an expert interviewer. Evaluate the candidate based on their CV and interview answers.

CV Context:
{cv_context}

Interview Questions and Answers:
{qa_pairs}

Return ONLY a JSON object with this exact structure, no extra text:
{{
    "final_score": <number 0-100>,
    "overall_summary": "<brief summary of candidate>",
    "per_question_feedback": [
        {{"question": "Q1", "feedback": "feedback text", "score": <0-10>}},
        {{"question": "Q2", "feedback": "feedback text", "score": <0-10>}},
        {{"question": "Q3", "feedback": "feedback text", "score": <0-10>}},
        {{"question": "Q4", "feedback": "feedback text", "score": <0-10>}},
        {{"question": "Q5", "feedback": "feedback text", "score": <0-10>}}
    ]
}}"""

        response = llm.predict(prompt)
        result = json.loads(response.strip())
        
        return JSONResponse(content={
            "signal": "evaluation_success",
            "project_id": str(project_id),
            "evaluation": result
        })
        
    except Exception as e:
        logger.error(f"Interview evaluation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "evaluation_error", "error": str(e)}
        )