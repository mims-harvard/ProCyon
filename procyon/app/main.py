import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger

# Import the key functions from the existing codebase
from procyon.inference.retrieval_utils import startup_retrieval, do_retrieval

app = FastAPI()

# Global variables to store model and device
model = None
device = None
data_args = None
all_protein_embeddings = None


class RetrievalRequest(BaseModel):
    task_desc: str = Field(description="The task description.")
    disease_desc: str = Field(description="The disease description.")
    instruction_source_dataset: str = Field(
        description="Dataset source for instructions - either 'disgenet' or 'omim'"
    )
    k: Optional[int] = Field(
        default=None,
        description="Number of top results to return. If None, returns all results",
        ge=1,
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the model and required components on startup"""
    global model, device, data_args, all_protein_embeddings

    if not os.getenv("HF_TOKEN"):
        raise EnvironmentError("HF_TOKEN environment variable not set")
    if not os.getenv("CHECKPOINT_PATH"):
        raise EnvironmentError("CHECKPOINT_PATH environment variable not set")
    if not os.getenv("HOME_DIR"):
        raise EnvironmentError("HOME_DIR environment variable not set")
    if not os.getenv("DATA_DIR"):
        raise EnvironmentError("DATA_DIR environment variable not set")
    if not os.getenv("LLAMA3_PATH"):
        raise EnvironmentError("LLAMA3_PATH environment variable not set")

    # Use the existing startup_retrieval function
    model, device, data_args, all_protein_embeddings = startup_retrieval(
        inference_bool=True
    )
    logger.info("Model loaded and ready")


@app.post("/retrieve")
async def retrieve_proteins(request: RetrievalRequest):
    """Endpoint to perform protein retrieval"""
    global model, device, data_args, all_protein_embeddings

    if not all([model, device, data_args, all_protein_embeddings]):
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Use the existing do_retrieval function
    results_df = do_retrieval(
        model=model,
        data_args=data_args,
        device=device,
        instruction_source_dataset=request.instruction_source_dataset,
        all_protein_embeddings=all_protein_embeddings,
        task_desc=request.task_desc,
        disease_desc=request.disease_desc,
    )

    results_df = results_df.fillna("")

    # Return all results if k is None, otherwise return top k
    if request.k is None:
        return {"results": results_df.to_dict(orient="records")}
    return {"results": results_df.head(request.k).to_dict(orient="records")}


if __name__ == "__main__":
    """
    This API endpoint will allow users to perform protein retrieval for a given disease description using the 
        pre-trained ProCyon model Procyon-Full.
    This API script can be run directly using the command `python main.py`
    this script will start the FastAPI server on port 8000
    The API will be available at http://localhost:8000
    An example request can be made using curl:
    curl -X POST "http://localhost:8000/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"task_desc": "Find proteins related to this disease", 
          "disease_desc": "Major depressive disorder",
          "instruction_source_dataset": "disgenet",
          "k": 1000}'
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
