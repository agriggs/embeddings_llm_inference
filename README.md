# Embeddings with LLM Inference

Code originally from this repo that does a great job describing embeddings [Embeddings and Vector Databases With ChromaDB](https://realpython.com/chromadb-vector-database/).

Updated to use multiple Large Language Models (LLMs) and removed the sample code that is not relevant.

## Contents

- [API Keys](#api-keys)
- [Python Environment](#python-environment)
- [Source Data](#source-data)
- [Model Configuration](#model-configuration)
- [Create the Embeddings (RAG Vectors)](#create-the-embeddings-rag-vectors)
- [Run GenAI Inference](#run-genai-inference)
- [Example Outputs](#example-outputs)

## API Keys
Get API keys for the providers you want to use and set their environment variables.
```
OPENAI_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
```

Get a Hugging Face token and set the environment variable. It is used to download the embedding model.
```
HF_TOKEN
```

Restart VS Code or other IDE so the environment variables are loaded.

## Python Environment

`uv` replaces `venv` and `pip`, and keeps a shared package cache.

Install `uv` by following the [getting started guide](https://docs.astral.sh/uv/getting-started/). It is only a few steps.

Create the virtual environment and activate it:

Windows (PowerShell)
```powershell
uv sync
.venv\Scripts\Activate.ps1
```

Linux / WSL
```bash
uv sync
source .venv/bin/activate
```

If you are behind a TLS intercept (corporate network), you may need to set these environment variables for Python HTTP libraries (`requests`, `httpx`).
```
REQUESTS_CA_BUNDLE
SSL_CERT_FILE
```

A corporate network may also block calls to some LLM API endpoints.

## Source Data

The car review source data is a set of CSV files in the [data/archive](./data/archive/) folder.

## Model Configuration

Edit the model IDs under `[models]` in [llm_models.toml](./llm_models.toml):

| Setting | Used for | Optional environment override |
| --- | --- | --- |
| `openai` | OpenAI inference | `OPENAI_MODEL` |
| `gemini` | Gemini inference | `GEMINI_MODEL` |
| `claude` | Claude inference | `ANTHROPIC_MODEL` |
| `embedding` | Creating and querying review embeddings | `EMBEDDING_MODEL` |

Environment variables take precedence over the file. API keys are configured
separately, as described above.

After changing the embedding model, run the create collection script below before inference.
Each model uses a separate collection, so existing embeddings are preserved.

## Create the Embeddings (RAG Vectors)

Embeddings are stored locally in ChromaDB under `car_review_embeddings`.
You only need to build them once per embedding model.

```
python create_car_review_collection.py
```

The scripts pick the best accelerator the installed PyTorch supports, fall back
to CPU, and print the selected device. Intel laptop NPUs would need a separate
backend such as OpenVINO; this project uses PyTorch.

On Windows, [pyproject.toml](./pyproject.toml) installs the CUDA 12.8 PyTorch build.
To confirm the GPU is enabled:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expect `+cu128 True`. `+cpu` or `False` means the CPU build is in use.

Build times:

| Hardware | Time |
| --- | --- |
| Intel Core Ultra 7 265U (CPU) | 8 minutes |
| Intel Core Ultra 7 265K (CPU) | 75 seconds |
| NVIDIA GeForce RTX 5080 (CUDA, Windows and WSL Ubuntu 24.04) | 10-20 seconds |

## Run GenAI Inference

OpenAI 
```
python llm_car_review_context.py openai
```

Gemini 
```
python llm_car_review_context.py gemini
```

Claude
```
python llm_car_review_context.py claude
```

## Example Outputs

```
Using claude model: claude-sonnet-5-5
Using embedding model: BAAI/bge-small-en-v1.5
Embedding device: cuda:0
```

### ChromaDB Semantic Output

#### Good Reviews (excerpt)

Make sure 'everything' works according to sales pitch and everything is there that you expect to be there, Shop around to locate the best deal and services., You get a lot for your money and great performance. We feel like this Has been a great buy for us., Satisfied with my purchase at Honda of Rockwall. All sales reps were very professional., Edmunds helped us feel confident with our purchase and gave us some help when dealing with the dealership., People doing reviews in this should be in the high end class like the Audi, Lexus, etc. Once you see the features and tech on par compare the price and save some money., High marks on comfort, options, materials and workmanship. Lacking a bit in performance and roominess. Overall, excellent!, Read reviews like Edmunds and Consumer Reports;  test drive several times; negoiatethe price et, It is best in class and max value for money. Just need to focus more on brand building from reliability POV., After all the facts and findings over all r

#### Worst Review (excerpt)

Purchased the car less than a year ago. Upside it is fun to drive. Downside is the dealer response to reported issues. Rear windows would not close completely and added to an already loud cruise experience. RH rear window fell inside of quarter panel. Nice. Took to Ft. Myers dealer, reported that and was fixed. Reported poor AC performance, not fixed, now it doesn’t work at all. Reported rear camera not working. Fixed, now it works sparratically, after all, I only need it in reverse :( Reported ambient temp read out inaccurate. Not fixed. Reported seat coolers very inefficient, service guy says they are OK. Speedometer drops off, dealer couldn’t duplicate, so it must be OK too. Seat memory system is forgetful, dealer couldn’t duplicate, so it must be OK. Tell that to my shorter than me wife. New one this week, dash dimmer switch activates all by itself, can hardly wait to listen to dealer on this one.

### Claude Inference Output

#### Summary of Good Reviews: Key Drivers of Customer Satisfaction

Based on these reviews, satisfaction comes down to a few themes.

##### 1. Value for money
Customers repeatedly say they feel they got more than they paid for:
- "You get a lot for your money and great performance."
- "Best in class and max value for money."
- One reviewer notes that features and tech rival luxury brands like Audi and Lexus at a lower price.

##### 2. Professional, trustworthy sales experience
- One buyer was satisfied because "all sales reps were very professional."
- Another said Edmunds gave them confidence and help when dealing with the dealership, which shows customers value support and transparency in the buying process.

##### 3. Quality, comfort, and features
- Reviewers gave high marks to "comfort, options, materials and workmanship."
- Delivering on features and technology builds loyalty.

##### 4. Delivering on the sales pitch
- Customers advise checking that "everything works according to the sales pitch" and that everything expected is there. Meeting expectations at delivery matters as much as the sale itself.

##### 5. Informed buyers
- Satisfied customers researched on Edmunds and Consumer Reports, test drove several times, shopped around, and negotiated. Dealerships that welcome this, with fair pricing and open test drives, tend to earn more trust.

##### Areas to improve
Even the positive reviews point to a few gaps:
- **Performance and roominess** were called "lacking a bit."
- **Brand reliability perception** needs more work, since one reviewer said the brand should "focus more on brand building from a reliability POV."

##### Bottom line
Great satisfaction comes from **fair value, professional and transparent service, quality that matches the pitch, and a buying process that lets customers feel informed and confident.**

#### Analysis of the Worst Review

**The Chevrolet Spark review is the most damaging.** The Subaru/Madison review is a close second.

##### Worst: the Spark owner
1. **The dealership dismissed safety concerns.** The owner reported unintended braking and a dangerous lack of power when pulling away from a stop. The dealership said it was "the way it's supposed to be" and "if I don't like it get a new car." If anyone is hurt, a documented pattern of brushing off safety complaints is a serious liability.
2. **The dealership refused to serve the customer.** After four visits for one issue, the owner says they were told to never come back. That reads as abandonment, not a service failure.
3. **It looks like a lemon-law case.** The owner reports repeated failed repairs (four visits for the power issue, three for the suspension, and a new rear squeak after the front was fixed), which could trigger legal and regulatory exposure.
4. **The customer was charged for a defect.** The owner says they were made to buy touch-up paint for paint that chips down to bare metal, which suggests a known defect was treated as customer wear.
5. **It asks other owners to pile on.** The review urges other Spark owners to post, which could amplify the damage.

##### Runner-up: the Madison Subaru review
This one is the worst for **trust and integrity**. The salesperson allegedly made a math error and then accused the customer of cheating. The review also describes repeated run-arounds, unresolved recall and electrical issues, and says outright to "avoid the west side dealership in Madison WI." It names a location and says the customer will escalate and never buy the brand again, so it is the most likely to cost us future sales.

##### Others
- **The first review** (Honda, Ft. Myers) shows a pattern of "could not duplicate" responses and failed repairs, plus the claim that the dealer has "a nasty view of customers." It is bad but less severe, and the issues are mostly comfort and convenience, not safety.
- **The Pilot review** is mostly about product defects, with relatively little blame on the dealership.

##### Caveat
The reviews name different brands and locations, so I can't tell which ones are about our store. If only some apply to us, we should confirm that first and prioritize outreach on those. For the Spark owner, I'd recommend immediate escalation to management and the manufacturer's regional rep, a documented safety inspection, and a personal call to the customer.