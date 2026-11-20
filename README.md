## FastAPI Application and Testing

### API Endpoints
- `GET /health`: Checks the API health and model loading status.
- `POST /predict`: Receives input data, validates it using Pydantic, and returns a prediction.

### Input Validation
- Used Pydantic schemas to validate incoming prediction requests.
- Rejects missing, invalid, and out-of-range input values.

### Testing
- Implemented 12 automated tests using Pytest.
- Tested health endpoint, model loading, prediction endpoint, and input validation.
- All 12 tests passed successfully.

### API Documentation
- Interactive API documentation is available through Swagger UI at `/docs`.

### Continuous Integration
- Configured GitHub Actions to install project dependencies and automatically run tests.
- Latest workflow run completed successfully.