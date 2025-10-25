# Flights by Country

A Python application that allows users to ask questions about flights for specific airports using the FlightAPI.io Airport Schedule API and LLM integration.

## Features
- Select from 6 major airports (DXB, LHR, CDG, SIN, HKG, AMS)
- Ask natural language questions about today's flights
- Get AI-generated answers based on real flight data
- Containerized with Docker for easy deployment

## Requirements
- Python 3.8+
- Docker (for containerization)
- FlightAPI.io API key
- OpenRouter API key (or other LLM provider)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   - `FLIGHT_API_KEY`=your_flightapi_key
   - `LLM_API_KEY`=your_openrouter_key
4. (Optional) Test FlightAPI connectivity: `python test_flight_api.py`
5. Run the application: `python app.py`

## Docker
Build and run with Docker:
```bash
docker build -t flights-by-country .
docker run -p 8000:8000 flights-by-country
```

## Architecture
The application follows a client-server architecture:
1. Frontend (HTML/CSS/JavaScript) sends requests to the backend
2. Backend (Python/FastAPI) processes requests
3. Flight data is retrieved from FlightAPI.io
4. LLM (via OpenRouter) interprets questions and generates answers
5. Responses are returned to the user

## API Endpoints
- `GET /` - Serve the frontend interface
- `POST /ask` - Process flight questions
- `POST /shutdown` - Shutdown the server gracefully

## LLM Integration Architecture
The LLM integration follows a context-aware approach:
1. Flight data is retrieved from FlightAPI.io for the selected airport
2. User questions are processed with the flight data as context
3. The LLM generates responses based on both the question and flight data
4. This approach ensures accurate, data-driven responses while maintaining natural language interaction

## Query Flow
1. User selects an airport and asks a question
2. Application fetches today's flight schedule for the selected airport
3. Flight data is formatted as context for the LLM
4. LLM processes the question with flight data context
5. Generated answer is returned to the user

## Design Choices
- **FastAPI**: Chosen for its high performance and built-in async support, enabling the application to handle 10+ requests per second
- **Async I/O**: Used throughout for non-blocking API calls to both FlightAPI and LLM providers
- **OpenRouter**: Selected as the LLM provider for access to multiple models with a single API key
- **Mistral 7B**: Used as the default model for its balance of performance and cost (free tier available)
- **Docker**: Containerization for easy deployment and scalability

## Limits
- **Flight Api**: 30 requests only per account
- **LLM**: Length of context, some info can be lost, because flight data too big, also limits per day
