# Embeddings with LLM Inference

Code from this overivew [Embeddings and Vector Databases With ChromaDB](https://realpython.com/chromadb-vector-database/). 

Updated to use multiple Large Language Models (LLMs) and removed the sample code that is not relevant.

## API Keys
Get API keys for the providers you want to use and set their environment variables.
```
OPENAI_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
```

Get Hugging Face Token and set the environment variable, used to download the embeddings model.
```
HF_TOKEN
```

Restart VS Code or other IDE so the environment variables are loaded.

## Python Environment

`uv` is used for Python `venv` and `pip` experience but a common cache for the packages.

Install and configure `uv` following instructions here, it is only a few steps: 
https://docs.astral.sh/uv/getting-started/

Create the virtual environment using `uv`

**Initialize the project with uv:**
```powershell
uv init --python 3.12
uv sync
. .venv/bin/activate
```

If running behind a TLS intercept (on corporate network), may need to set these environment variables for Python libs: requests, https.
```
REQUESTS_CA_BUNDLE
SSL_CERT_FILE
```

Corporate network may also block calls to some of the LLM AI endpoints.

## Source data

Car reviews source data is a bunch of CSV in [data/archive](./data/archive/) folder.

## Model configuration

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
Each model uses a separate collection, preserving existing embeddings.

## Create the embeddings (RAG vectors)

Embeddings are stored locally in ChromaDB under `car_review_embeddings`.

The scripts automatically select an accelerator supported by the installed
PyTorch runtime, falling back to CPU, and print the selected device.
GPU use requires an appropriate PyTorch build and drivers. Intel laptop NPUs
require a separate backend such as OpenVINO; this project currently uses PyTorch.

On Intel Core Ultra 7 265U 2.10 GHz embeddings took 8 minutes to build and store. On GeForce RTX 5080 with NVIDAI Cuda took 21 seconds. 

Only need to build once for each embedding model. 

```
python create_car_review_collection.py
```

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

The outputs below are historical examples from earlier model versions.

### ChromaDB Semantic Output

<b>Good reviews</b>

Great value, awesome reliability, I normally dont do reviews, but thought this was worth writing. I own 3 pickups and am not brand dedicated because there are so many nice trucks. I own 3 different trucks. I am drawn to looks first then power and interior. I have notice reading reviews that most want to beat the trucks up, there are no perfect trucks. The Nissan drew me in because it was different and I dont recall any recalls. I owned a Xterra and it was bulletproof and am hoping for the same on my Titan. After looking at all brands and I liked them all, I really like the warranty on my Titan and I got more equipment for the money than the competitors. My first and second tank of fuel at 70 mpg I got 17.5, I was very pleased and when it get broke in I hope for 20. I find it to be very quiet and so were all the others on top of the line. I guess I could beat the truck up with small complaints but sure makes spending my money on it hard to swallow. If the truck last me for 10 years I will be very pleased. I know the ride will get much better as it has in all my cars and trucks in time. I hope you enjoy your new purchase as much as I am enjoying mine., Very  pleased! Great deal! So far so good!, Excellent quality, technology, comfort and value. Way impressed with this vehicle., Things I love: styling, performance, large info center, smooth acceleration, no gas stations, low maintenance costs, incredible sound system, frunk and sub-trunk, rear cargo space, HOV lane, federal tax credit, summons feature, replacement parts seem nicely priced, my first Service Center experience was excellent, my first body shop experience was very good, many people think you are "cool" and environmentally friendly (and you are).          Things I wish were better: fit and finish of body parts (not up to premium car standard), auto pilot still not a wow...needs more development, blind spot monitor is below expectations for such an advance vehicle, difficulty getting in and out of front seat...if you are tall, the front seat goes behind the "B" pillar too far thus the entering and exiting is more difficult. Final Verdict. I love the car. I doubt I would ever buy an ICE car again., Good overall but a lot of others out there., Excellent choice, so far, so good.  I got the S-Line, Premium Plus package and am very happy with it.  Quality it top notch inside and out.  No issues at all, though I've only had it for a month and about 1500 miles., After all the facts and findings over all review is outstanding., Excellent features and performance!  

<b>Worst reviews:</b> 

 I have been to the dealership four times and I still have unresolved electrical issues, from the Bluetooth, backup camera, truck wont open, black screen every other trip turning off, clock wont work, seatbelts NOT WORKING !!!!!!!! which really pisses me off as I have a child in the car. So many issues in a short period of time it has become the worst vehicle I have owned unfortunately and to the point where I will drop the car off back at the dealership and purchase something else. I don't have the time to deal with a vehicle of 53k value as if it were pieced together from spare parts from a junkyard. I always read reviews before buying vehicles and I have stuck with Nissan & Infiniti for the past ten years for their workhorse engines as two of my Nissans went to 289k & 377k miles but after this SUV I am jumping ship as soon as possible.

 ### Claude Inference Output

<b>Claude generated summary with claude-sonnet-4-5 of good reviews:</b> 
Based on the detailed positive reviews provided, the keys to great customer satisfaction are:

#### 1. **Value for Money**
Customers emphasize getting "more equipment for the money than competitors" and appreciate "great deals." The perception of receiving excellent features at a competitive price point is crucial.

#### 2. **Reliability and Quality**
Multiple reviews mention "awesome reliability," "bulletproof" performance, and "excellent quality." Customers want vehicles they can trust for the long term (10+ years).

#### 3. **Strong Warranty and Low Maintenance**
A generous warranty package and "low maintenance costs" provide peace of mind and reduce long-term ownership concerns.

#### 4. **Performance and Technology**
Customers value:
- "Smooth acceleration" and performance
- Advanced technology features (large info centers, autopilot, summons features)
- Modern amenities (sound systems, cargo space)

#### 5. **Unique Features and Differentiation**
Buyers appreciate vehicles that are "different" and stand out, whether through styling, environmental benefits (HOV lanes, no gas stations), or innovative features.

#### 6. **Positive Ownership Experience**
- Good service center experiences
- Fair replacement part pricing
- Meeting or exceeding expectations (fuel economy, comfort)

**Bottom Line**: Great customer satisfaction comes from delivering reliable, feature-rich vehicles at competitive prices, backed by strong warranties and excellent service—essentially exceeding value expectations while minimizing ownership hassles.

<b>Claude generated summary with claude-sonnet-4-5 of the single worst review:</b>

Based on these reviews, **the second review (about the GMC vehicle with transmission issues)** has the worst implications for the dealership, despite the reviewer noting the dealer has been "very good to work with."

Here's why this is most damaging:

#### Critical Safety and Operations Issues:

1. **Repeated, Extended Downtime**: The vehicle has been in the shop for 14 days across 3 occasions and is STILL there. This suggests:
   - Inability to diagnose problems correctly the first time
   - Poor parts inventory management
   - Inefficient service operations

2. **Parts Availability Crisis**: The dealership didn't have basic parts on hand for multiple repairs:
   - Transmission leak parts
   - Spindle nut (which was broken by the mechanic)
   - Rear shock and spring bushing components

   This reveals systematic inventory problems that extend repair times unnecessarily.

3. **Mechanic Error**: Breaking the spindle nut during reassembly shows potential technician competency issues or rushing through repairs.

4. **Factory Defect Not Caught**: The rear wheel alignment issue that wore tires to the cord should have been caught during pre-delivery inspection (PDI). This is a fundamental dealership failure. 

5. **Potential Lemon**: The customer is explicitly considering this a lemon, which could lead to legal action involving the dealership.

#### Why This is Worse Than the Others:

- **Review 1 (Nissan/Infiniti)**: Focuses on vehicle quality issues, not dealership service failures
- **Reviews 3, 4, 5**: Focus on vehicle design/comfort complaints, not dealership operations

The GMC review reveals operational incompetence that reflects directly on YOUR dealership's service department, parts management, and quality control processes.
