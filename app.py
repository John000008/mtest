from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import aiohttp
from dotenv import load_dotenv
import uvicorn
import json
import os
import re
import logging

# Load environment variables
load_dotenv()

# Configuration
FLIGHT_API_KEY = os.getenv("FLIGHT_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
FLIGHT_API_URL = "https://api.flightapi.io/schedule"
AIRPORTS = ["DXB", "LHR", "CDG", "SIN", "HKG", "AMS"]


class QuestionRequest(BaseModel):
    airport: str
    question: str


class FlightDataResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    user_airport: Optional[str] = None


class FlightDataProcessor:
    """Handles fetching and processing flight data from the API."""

    def __init__(self, api_key: str, api_url: str, airports: list):
        self.api_key = api_key
        self.api_url = api_url
        self.airports = airports
        # get logger
        self.logger = logging.getLogger('main')
        self.file_path = 'flight_data_last.json'

    async def read_json(self, airport: str) -> FlightDataResponse:
        data = {}
        with open(self.file_path, "r") as file:
            data = json.load(file)
        self.logger.debug(f"Data loaded from {self.file_path}: data: {data}")
        return FlightDataResponse(success=True, data=data, user_airport=airport)

    async def fetch_flight_data(self, airport: str) -> FlightDataResponse:
        """Fetch flight schedule data for the specified airport."""
        if not self.api_key:
            return FlightDataResponse(success=False, error="Flight API key not configured")

        try:
            # Using aiohttp for async requests
            async with aiohttp.ClientSession() as session:
                # Simplified approach matching user's working example
                url = f"{self.api_url}/{self.api_key}?"
                params = {
                    "mode": "arrivals",  # Changed to arrivals to match requirements
                    "iata": airport,  # Use the selected airport
                    "day": 0  # 0 for today
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.info(f"Flight API Response:")
                        self.logger.info(f"  Status: {response.status}")
                        self.logger.info(f"  Data type: {type(data)}")
                        self.logger.info(f"  Data raw: {data}")
                        self.logger.info(f"  Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        # Dump flight data to a local JSON file for debugging
                        with open(self.file_path, "w") as f:
                            json.dump(data, f, indent=2)

                        self.logger.info(f"Flight data dumped to: {self.file_path}")

                        return FlightDataResponse(success=True, data=data, user_airport=airport)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Flight API Error:")
                        self.logger.error(f"  Status: {response.status}")
                        self.logger.error(f"  Error text: {error_text}")

                        # Handle specific error cases
                        if response.status == 403:
                            if "maximum limits" in error_text.lower():
                                user_error = "FlightAPI quota limit reached. Please upgrade your account or wait for quota reset."
                            else:
                                user_error = "Access denied to FlightAPI. Please check your API key and account status."
                        elif response.status == 401:
                            user_error = "Invalid FlightAPI key. Please check your API key configuration."
                        elif response.status == 404:
                            user_error = "Flight data not found. Please check the airport code and try again."
                        else:
                            user_error = f"FlightAPI request failed with status {response.status}. Please try again later."

                        return FlightDataResponse(
                            success=False,
                            error=user_error
                        )
        except Exception as e:
            return FlightDataResponse(success=False, error=str(e))


class LLMQueryHandler:
    """Handles querying the LLM with flight data and user questions."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        # get logger
        self.logger = logging.getLogger('main')


    async def query_llm(self, prompt: str, flight_data: Dict[str, Any], user_airport: str) -> str:
        """Query LLM to interpret the question and generate an answer based on flight data."""
        if not self.api_key:
            return "LLM API key not configured"

        try:
            # Using OpenRouter as LLM provider
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            context = flight_data['airport']['pluginData']

            system_message = """You are a helpful assistant that answers questions about flight schedules. 
Use the provided flight data to answer the user's question accurately. 
Focus only on today's flights. If the data is truncated or incomplete, mention that in your response.
Provide clear, concise answers based on the flight information provided. Remove any tags such as [/B_INST] or any other from your answer in your 'content' key """

            user_message = f"""Flight Information from user {user_airport} airport:
{context}

Question: {prompt}

Please provide a clear and accurate answer based on the flight information above."""
            model_info = "meta-llama/llama-3.3-70b-instruct:free"
            # model_info = "mistralai/mistral-7b-instruct:free"
            payload = {
                "model": model_info,
                "transforms": ["middle-out"],
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "temperature": 0.7
            }

            # Add debugging information
            self.logger.info(f"Sending request to LLM: {self.url}")
            self.logger.info(f"Headers: {headers}")
            self.logger.info(f"Payload size: {len(json.dumps(payload))} characters")
            self.logger.info(f"Context size: {len(context)} characters")
            self.logger.info(f"Full payload: {json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload) as response:
                    self.logger.info(f"LLM response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"LLM response: {result}")
                        # Clean up the response content to remove special tokens
                        content = result["choices"][0]["message"]["content"]
                        # Remove common special tokens from Mistral models
                        content = self.clean_result(content)
                        # Remove any leading/trailing whitespace
                        content = content.strip()
                        return content
                    else:
                        error_text = await response.text()
                        self.logger.error(f"LLM error response: {error_text}")
                        # Try a fallback model if the current one fails
                        if "mistral" in payload["model"]:
                            self.logger.error("Trying fallback model...")
                            payload["model"] = "openchat/openchat-7b:free"
                            async with session.post(self.url, headers=headers, json=payload) as fallback_response:
                                self.logger.info(f"Fallback LLM response status: {fallback_response.status}")
                                if fallback_response.status == 200:
                                    fallback_result = await fallback_response.json()
                                    self.logger.info(f"Fallback LLM response: {fallback_result}")
                                    # Clean up the response content to remove special tokens
                                    content = fallback_result["choices"][0]["message"]["content"]
                                    # Remove common special tokens from Mistral models
                                    content = self.clean_result(content)
                                    # Remove any leading/trailing whitespace
                                    content = content.strip()
                                    return content
                                else:
                                    fallback_error = await fallback_response.text()
                                    self.logger.error(f"Fallback LLM error: {fallback_error}")
                                    return f"LLM request failed with status {response.status}: {error_text}"
                        return f"LLM request failed with status {response.status}: {error_text}"
        except Exception as e:
            self.logger.error(f"Exception in LLM query: {str(e)}")
            return f"Error querying LLM: {str(e)}"

    def clean_result(self, content):
        # clean up from any tags []
        try:
            content = re.sub(r'\[.*?\]', '', content)
        except Exception as e:
            self.logger.error(f"Exception in cleaning result: {content}: {e}")
        return content


class FlightApp:
    """Main application class that orchestrates the flight query system."""

    def __init__(self):
        # get logger
        self.logger = logging.getLogger('main')

        self.flight_processor = FlightDataProcessor(FLIGHT_API_KEY, FLIGHT_API_URL, AIRPORTS)
        self.llm_handler = LLMQueryHandler(LLM_API_KEY)
        self.app = FastAPI(title="Flights by Country")
        self.setup_routes()

    def setup_routes(self):
        """Set up the FastAPI routes."""
        self.app.get("/", response_class=HTMLResponse)(self.read_root)
        self.app.post("/ask")(self.ask_question)

    async def read_root(self):
        """Serve the main HTML interface."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flights by Country</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                .form-group {
                    margin-bottom: 20px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                select, input[type="text"] {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                button {
                    background-color: #007bff;
                    color: white;
                    padding: 12px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    width: 100%;
                }
                button:hover {
                    background-color: #0056b3;
                }
                #result {
                    margin-top: 20px;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #e9ecef;
                    display: none;
                }
                .loading {
                    text-align: center;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Flights by Country</h1>
                <form id="questionForm">
                    <div class="form-group">
                        <label for="airport">Select Airport:</label>
                        <select id="airport" name="airport" required>
                            <option value="">-- Select an Airport --</option>
                            <option value="DXB">DXB - Dubai International Airport</option>
                            <option value="LHR">LHR - London Heathrow Airport</option>
                            <option value="CDG">CDG - Paris Charles de Gaulle Airport</option>
                            <option value="SIN">SIN - Singapore Changi Airport</option>
                            <option value="HKG">HKG - Hong Kong International Airport</option>
                            <option value="AMS">AMS - Amsterdam Schiphol Airport</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="question">Ask a Question:</label>
                        <input type="text" id="question" name="question" placeholder="e.g., How many flights arrived from Germany?" required>
                    </div>
                    <button type="submit">Get Answer</button>
                </form>
                <div id="result"></div>
            </div>
            <script>
                document.getElementById('questionForm').addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    const airport = document.getElementById('airport').value;
                    const question = document.getElementById('question').value;
                    const resultDiv = document.getElementById('result');
                    
                    if (!airport || !question) {
                        alert('Please select an airport and enter a question');
                        return;
                    }
                    
                    // Show loading message
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = '<div class="loading">Processing your question...</div>';
                    
                    try {
                        const response = await fetch('/ask', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ airport, question })
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            resultDiv.innerHTML = `<h3>Answer:</h3><p>${data.answer}</p>`;
                        } else {
                            // Check if we have an error field or fall back to detail
                            const errorMessage = data.error || data.detail || 'An unknown error occurred';
                            resultDiv.innerHTML = `<h3>Error:</h3><p>${errorMessage}</p>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `<h3>Error:</h3><p>Failed to process your request: ${error.message}</p>`;
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    async def ask_question(self, request: QuestionRequest):
        """Process a question about flights for a specific airport."""
        # Fetch flight data
        flight_response = await self.flight_processor.fetch_flight_data(request.airport)
        # for debug
        # flight_response = await self.flight_processor.read_json(request.airport)

        if not flight_response.success:
            # Log the error for debugging
            self.logger.error(f"Flight data fetch failed: {flight_response.error}")
            # Return a more informative error response instead of raising HTTPException
            return {"error": flight_response.error,
                    "answer": "Sorry, I couldn't fetch flight data. Please check your API key and try again."}

        # Query LLM with flight data and question
        answer = await self.llm_handler.query_llm(request.question, flight_response.data, flight_response.user_airport)
        return {"answer": answer}


class LoggerCLS:

    def __init__(self):
        self.log_file_name = f'latest.log'
        self.logger = None

    def start(self):
        self.logger = self._init_logging()

    def _init_logging(self):
        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)

        stream_h = logging.StreamHandler()
        stream_h.setLevel(logging.INFO)
        stream_h_format = logging.Formatter('[%(asctime)s]: %(message)s')
        stream_h.setFormatter(stream_h_format)
        logger.addHandler(stream_h)

        file_h = logging.FileHandler(self.log_file_name, encoding='utf-8')
        file_h.setLevel(logging.DEBUG)
        file_h_format = logging.Formatter('[%(name)s:%(levelname)s:%(asctime)s] : %(message)s')

        file_h.setFormatter(file_h_format)
        logger.addHandler(file_h)

        return logger

# Initialize the application
init_logger = LoggerCLS()
init_logger.start()
logger = init_logger.logger

flight_app = FlightApp()
app = flight_app.app

if __name__ == "__main__":
    # Run the server
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.error("Server interrupted")
    finally:
        logger.info("Server stopped")
