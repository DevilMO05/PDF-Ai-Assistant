import streamlit as st
import torch
import faiss

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF AI Assistant")
st.write("Upload a PDF, ask questions, and get answers using Mistral + FAISS.")


# =========================================================
# Load Mistral Model
# =========================================================

@st.cache_resource
def load_llm():

    modelName = "mistralai/Mistral-Nemo-Instruct-2407"

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# =========================================================
# Load Embedding Model
# =========================================================

@st.cache_resource
def load_embedding_model():

    modelName = "sentence-transformers/all-MiniLM-L6-v2"

    model = SentenceTransformer(modelName)

    return model


# =========================================================
# PDF Functions
# =========================================================

def extractPDF(pdf_file):

    reader = PdfReader(pdf_file)

    fullText = ""

    for page in reader.pages:

        pageText = page.extract_text()

        if pageText:
            fullText += pageText + "\\n"

    return fullText


def chunkText(text, chunkSize=50, overlap=5):

    words = text.split()

    chunks = []

    step = chunkSize - overlap

    for i in range(0, len(words), step):

        chunk = " ".join(words[i:i + chunkSize])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =========================================================
# Embeddings + FAISS
# =========================================================

def embedChunks(chunks, model):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings


def createFaiss_i(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index


def searchIndex(query, model, index, chunks, k=3):

    queryEmbedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        queryEmbedding,
        k
    )

    results = []

    for i in indices[0]:

        if i != -1:
            results.append(chunks[i])

    return results


# =========================================================
# Generate Answer
# =========================================================

def generateText(
    prompt,
    tokenizer,
    model,
    max_length=700
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"

    )

    # Move input tensors to the same device as the model
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("Settings")

    topK = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    chunkSize = st.number_input(
        "Chunk size",
        min_value=20,
        max_value=500,
        value=50
    )

    overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=100,
        value=5
    )

    maxLength = st.number_input(
        "Maximum answer length",
        min_value=100,
        max_value=2000,
        value=700
    )


# =========================================================
# Load Models
# =========================================================

with st.spinner("Loading models..."):

    tokenizer, llm = load_llm()

    embeddingModel = load_embedding_model()


# =========================================================
# PDF Upload
# =========================================================

st.header("Upload PDF")

pdfFile = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# =========================================================
# Process PDF
# =========================================================

if pdfFile is not None:

    st.success(f"Uploaded: {pdfFile.name}")

    if st.button(
        "Process PDF",
        type="primary"
    ):

        with st.spinner("Reading PDF..."):

            text = extractPDF(pdfFile)

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

        else:

            with st.spinner("Creating chunks..."):

                chunks = chunkText(
                    text,
                    chunkSize=chunkSize,
                    overlap=overlap
                )

            with st.spinner("Creating embeddings..."):

                embeddings = embedChunks(
                    chunks,
                    embeddingModel
                )

            with st.spinner("Creating FAISS index..."):

                index = createFaiss_i(
                    embeddings
                )

            # Save everything in Streamlit session
            st.session_state["text"] = text
            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = embeddings
            st.session_state["index"] = index

            st.success(
                f"PDF processed successfully! "
                f"Created {len(chunks)} chunks."
            )


# =========================================================
# Question Section
# =========================================================

if "index" in st.session_state:

    st.divider()

    st.header("Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: Who is mentioned in the document?"
    )

    askButton = st.button(
        "Ask",
        type="primary"
    )

    if askButton and question.strip():

        chunks = st.session_state["chunks"]

        index = st.session_state["index"]

        # ---------------------------------------------
        # Retrieve relevant chunks
        # ---------------------------------------------

        with st.spinner("Searching the PDF..."):

            top_chunks = searchIndex(
                question,
                embeddingModel,
                index,
                chunks,
                k=topK
            )

        # ---------------------------------------------
        # Build prompt
        # ---------------------------------------------

        context = "\\n\\n".join(
            top_chunks
        )

        prompt = '''Answer the question using only the provided context.

Question:
{}

Context:
{}

Answer:'''.format(question, context)

        # ---------------------------------------------
        # Generate answer
        # ---------------------------------------------

        with st.spinner("Generating answer..."):

            answer = generateText(
                prompt,
                tokenizer,
                llm,
                max_length=maxLength
            )

        # Remove prompt from generated text if present
        if "Answer:" in answer:

            answer = answer.split(
                "Answer:",
                1
            )[1].strip()

        # ---------------------------------------------
        # Display Answer
        # ---------------------------------------------

        st.header("Answer")

        st.write(answer)

        # ---------------------------------------------
        # Display Retrieved Chunks
        # ---------------------------------------------

        with st.expander(
            "📚 Retrieved Context"
        ):

            for i, chunk in enumerate(
                top_chunks,
                1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(chunk)

                st.divider()


# =========================================================
# Initial Instructions
# =========================================================

else:

    st.info(
        "👆 Upload a PDF and click **Process PDF** "
        "to start asking questions."
    )