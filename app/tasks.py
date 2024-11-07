import time
from celery_config import celery
from ocr_strategies.marker import MarkerOCRStrategy
from ocr_strategies.tesseract import TesseractOCRStrategy
import redis
import os
import ollama

OCR_STRATEGIES = {
    'marker': MarkerOCRStrategy(),
    'tesseract': TesseractOCRStrategy()
}

# Connect to Redis
redis_url = os.getenv('REDIS_CACHE_URL', 'redis://redis:6379/1')
redis_client = redis.StrictRedis.from_url(redis_url)

@celery.task(bind=True)
def ocr_task(self, pdf_bytes, strategy_name, pdf_hash, ocr_cache, prompt, model):
    """
    Celery task to perform OCR processing on a PDF file.
    """
    if strategy_name not in OCR_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: marker, tesseract")
    
    start_time = time.time()

    ocr_strategy = OCR_STRATEGIES[strategy_name]
    self.update_state(state='PROGRESS', status="File uploaded successfully", meta={'progress': 10})  # Example progress update
    
    extracted_text = None
    if ocr_cache:
        cached_result = redis_client.get(pdf_hash)
        if cached_result:
            # Return cached result if available
            extracted_text = cached_result.decode('utf-8')

    if extracted_text is None:
        print("Extracting text from PDF...")
        elapsed_time = time.time() - start_time
        self.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Extracting text from PDF', 'start_time': start_time, 'elapsed_time': time.time() - start_time})  # Example progress update
        extracted_text = ocr_strategy.extract_text_from_pdf(pdf_bytes)
    else:
        print("Using cached result...")

    print("Extracted text: " + extracted_text)
    self.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Text extracted', 'extracted_text': extracted_text, 'start_time': start_time, 'elapsed_time': time.time() - start_time})  # Example progress update

    if ocr_cache:
        redis_client.set(pdf_hash, extracted_text)

    if prompt:
        print("Transforming text using LLM (prompt={prompt}, model={model}) ...")
        self.update_state(state='PROGRESS', meta={'progress': 75, 'status': 'Processing LLM', 'start_time': start_time, 'elapsed_time': time.time() - start_time})  # Example progress update
        llm_resp = ollama.generate(model, prompt + extracted_text, stream=True)
        num_chunk = 1
        for chunk in llm_resp:
            self.update_state(state='PROGRESS', meta={'progress': num_chunk , 'status': 'LLM Processing chunk no: ' + str(num_chunk), 'start_time': start_time, 'elapsed_time': time.time() - start_time})  # Example progress update
            num_chunk += 1
            extracted_text += chunk['response']

    # Optional call to generate tags and summary
    if prompt and model:
        tags_summary = generate_tags_summary(prompt, model)
        extracted_text += "\n\nTags and Summary:\n" + tags_summary

    self.update_state(state='DONE', meta={'progress': 100 , 'status': 'Processing done!', 'start_time': start_time, 'elapsed_time': time.time() - start_time})  # Example progress update

    return extracted_text

def generate_tags_summary(prompt, model):
    """
    Function to generate tags and summary using the LLM.
    """
    try:
        response = ollama.generate(model, prompt)
    except ollama.ResponseError as e:
        print('Error:', e.error)
        if e.status_code == 404:
            print("Error: ", e.error)
            ollama.pull(model)
        raise Exception("Failed to generate tags and summary with Ollama API")

    generated_text = response.get("response", "")
    return generated_text
