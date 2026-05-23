from fastapi import APIRouter, UploadFile, Request, status
from fastapi.responses import JSONResponse
from uuid import UUID
import fitz
import json
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])

        settings = get_settings()

        embeddings = CohereEmbeddings(
            cohere_api_key=settings.COHERE_API_KEY,
            model="embed-multilingual-v3.0"
        )

        collection_name = str(project_id).replace("-", "_")

        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=settings.POSTGRES_URL,
            pre_delete_collection=True,
        )

        llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_ID,
            google_api_key=settings.GEMINI_API_KEY,
            convert_system_message_to_human=True,
        )

        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        result = qa.run(
            "Based on this CV, generate exactly 5 technical interview questions. "
            "Return ONLY a JSON array of 5 objects, no extra text. Each object must have exactly two fields: "
            "'QuestionText' (the question) and 'ExpectedKeyPoints' (comma-separated key points the answer should cover). "
            "Example: [{\"QuestionText\": \"Q1?\", \"ExpectedKeyPoints\": \"point1, point2, point3\"}, ...]"
        )
        clean_result = result.strip()
        if clean_result.startswith("```"):
            clean_result = clean_result.split("```")[1]
            if clean_result.startswith("json"):
                clean_result = clean_result[4:]
        clean_result = clean_result.strip()
        questions = json.loads(clean_result)

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

        qa_pairs = "\n".join([
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(eval_request.questions, eval_request.answers))
        ])

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        cv_docs = retriever.get_relevant_documents(qa_pairs)
        cv_context = "\n".join([doc.page_content for doc in cv_docs])

        llm = ChatGoogleGenerativeAI(
            model=settings.GENERATION_MODEL_ID,
            google_api_key=settings.GEMINI_API_KEY,
            convert_system_message_to_human=True,
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
        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
            if clean_response.startswith("json"):
                clean_response = clean_response[4:]
        clean_response = clean_response.strip()
        result = json.loads(clean_response)

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