i now need to do a RAG benchmark.
i would need to find retrieval latency of the rag engine.
i need to then augment the prompt questions about resume.pdf with the retrieved RAG data
i need to send this to an LLM and compare it with the gold answer (do semantic similarity for comparison)

i need this to be run on at least llama3b model. 

1. you yourself generate a dataset.json in this current dir to contain questions and answer pairs (10) relavant to the resume.

2. run the rag engine on each question. measure retrieval latency, carbon emission etc...

3. get the retrieved chunks and augment the question prompt and run inference usign llama.cpp just like the other benchmark scripts (qa.py, summarization.py)

4. collect the same metrics as in qna. 

5. measure the semantic similarity with the real answers.

6. finally, condense all data and information into a summary.json