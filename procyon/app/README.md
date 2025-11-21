# Locally hosted API for ProCyon in protein retrieval mode


This API endpoint will allow users to perform protein retrieval for a given disease description using the 
    pre-trained ProCyon model Procyon-Full.

This API script can be run directly using the command `python main.py`

This script will start the FastAPI server on port 8000

The API will be available at http://localhost:8000

An example request can be made using `curl`:

```
curl -X POST "http://localhost:8000/retrieve" \
 -H "Content-Type: application/json" \
 -d '{"task_desc": "Find proteins related to this disease", 
      "disease_desc": "Major depressive disorder",
      "instruction_source_dataset": "disgenet",
      "k": 1000}'
```
