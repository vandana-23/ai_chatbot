from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .rag_model import get_qa_chain
import logging
from django.http import HttpResponse

logger = logging.getLogger(__name__)

qa_chain = None

def home(request):
    return HttpResponse("Hello — homepage is working.")

@api_view(["GET", "POST"])
def chat_response(request):
    global qa_chain

    if qa_chain is None:
        try:
            qa_chain = get_qa_chain()
        except Exception as e:
            logger.exception("Failed to build QA chain")
            return Response({"error": "Server failed to initialize QA chain"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    if request.method == "POST":
        user_message = request.data.get("message") or request.data.get("question") or ""
    else:
        user_message = request.query_params.get("q") or "what is goklyn?"

    user_message = (user_message or "").strip()
    if not user_message:
        return Response({"error": "No message provided."}, status=status.HTTP_400_BAD_REQUEST)


    if len(user_message) > 2000:
        return Response({"error": "Message too long"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        retrieved_docs = qa_chain.retriever.get_relevant_documents(user_message)
        logger.info(f"Retrieved {len(retrieved_docs)} docs")
        for i, d in enumerate(retrieved_docs, 1):
             logger.info(f"Doc {i}: {d.page_content[:300]}")

        result = qa_chain.invoke({"query": user_message})
    except Exception as e:
        logger.exception("Chain execution failed")
        return Response({"error": f"Chain execution failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
    answer = result.get("result") or result.get("answer") or result.get("output_text") or ""
    source_docs = result.get("source_documents", []) or []

    sources = []
    for doc in source_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        sources.append({
            "source": metadata.get("source", "unknown"),
            "snippet": (doc.page_content or "")[:400].strip()
        })

    response_payload = {
        "query": user_message,
        "reply": answer,
        # "sources": sources  
    }

    return Response(response_payload, status=status.HTTP_200_OK)

