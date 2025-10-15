import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents

client = px.Client(project_name="mw-rag")

eval_model = OpenAIModel(model="gpt-4")

relevance_eval = RelevanceEvaluator(eval_model)
hallucination_eval = HallucinationEvaluator(eval_model)
qa_eval = QAEvaluator(eval_model)

retrieved_docs_df = get_retrieved_documents(client)
queries_df = get_qa_with_reference(client)

if retrieved_docs_df is None or queries_df is None:
    raise ValueError("Failed to retrieve data from Phoenix client")

# Run relevance on retriever docs
retrieved_relevance_df = run_evals(
    evaluators=[relevance_eval],
    dataframe=retrieved_docs_df,
    provide_explanation=True,
    concurrency=20,
)[0]

# Run hallucination + QA correctness
hallucination_df, qa_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_eval, qa_eval],
    provide_explanation=True,
    concurrency=20,
)
