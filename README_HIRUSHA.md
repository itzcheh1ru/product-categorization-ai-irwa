# Hirusha's Components - Product Categorization AI

## Overview
This document describes the components implemented by Hirusha (IT23426580) for the Product Categorization AI project.

## Assigned Components

### 1. TagGeneratorAgent (`backend/agents/tag_generator_agent.py`)
**Purpose**: Generates relevant tags for products using multiple approaches.

**Features**:
- **LLM-based tag generation**: Uses language models for intelligent tag creation
- **NLP-based tag extraction**: Extracts nouns and adjectives from product names
- **Rule-based tag generation**: Creates tags based on product attributes (color, season, usage, etc.)
- **Tag deduplication**: Removes duplicate tags and keeps highest confidence versions
- **Comprehensive error handling**: Graceful fallbacks when components are unavailable
- **Configurable**: Supports optional LLM and NLP processors

**Key Methods**:
- `generate_tags(product_data, attributes)`: Main method for tag generation
- `_generate_llm_tags()`: LLM-based tag generation
- `_generate_nlp_tags()`: NLP-based tag extraction
- `_generate_rule_based_tags()`: Rule-based tag creation
- `clean_tag()`: Tag text cleaning and formatting

**Example Usage**:
```python
from agents.tag_generator_agent import TagGeneratorAgent

agent = TagGeneratorAgent()
product_data = {
    'productDisplayName': 'Red Cotton Summer Dress',
    'baseColour': 'Red',
    'usage': 'Casual',
    'articleType': 'Dress',
    'season': 'Summer'
}

result = agent.generate_tags(product_data)
print(f"Generated {len(result['tags'])} tags")
```

### 2. Security Module (`backend/core/security.py`)
**Purpose**: Comprehensive security utilities for authentication, authorization, and data protection.

**Features**:
- **Password Management**: Secure hashing with bcrypt, password strength validation
- **JWT Token Handling**: Access and refresh token creation/validation
- **Input Sanitization**: Protection against XSS and injection attacks
- **CSRF Protection**: Token generation and verification
- **File Safety**: Filename validation to prevent path traversal
- **Rate Limiting**: Basic rate limiting functionality
- **Data Hashing**: Secure hashing for sensitive data

**Key Functions**:
- `verify_password()`, `get_password_hash()`: Password management
- `validate_password_strength()`: Password strength validation with detailed feedback
- `create_access_token()`, `create_refresh_token()`: JWT token creation
- `decode_access_token()`, `decode_refresh_token()`: JWT token validation
- `sanitize_input()`: Input sanitization
- `generate_csrf_token()`, `verify_csrf_token()`: CSRF protection
- `is_safe_filename()`: Filename safety validation

**Example Usage**:
```python
from core.security import get_password_hash, verify_password, validate_password_strength

# Password hashing
password = "SecurePassword123!"
hashed = get_password_hash(password)
is_valid = verify_password(password, hashed)

# Password validation
validation = validate_password_strength(password)
print(f"Password score: {validation['score']}/5")
```

### 3. API Schemas (`backend/api/schemas.py`)
**Purpose**: Comprehensive Pydantic models for API data validation and serialization.

**Features**:
- **Product Data Models**: Complete product information schemas
- **Tag Models**: Tag structure with confidence scores and sources
- **Request/Response Models**: API request and response schemas
- **Error Handling**: Comprehensive error response models
- **Validation**: Built-in data validation with custom validators
- **Documentation**: Detailed field descriptions and examples

**Key Models**:
- `ProductData`: Product input data schema
- `Tag`: Individual tag schema with confidence and source
- `TagGenerationRequest`: Tag generation API request
- `TagGenerationResponse`: Tag generation API response
- `ErrorResponse`: Error response schema
- `HealthCheck`: Health check response schema
- `ValidationError`: Validation error details

**Example Usage**:
```python
from api.schemas import ProductData, Tag, TagGenerationRequest

# Create product data
product = ProductData(
    description="Red Cotton Summer Dress",
    base_colour="Red",
    article_type="Dress"
)

# Create tag
tag = Tag(
    tag="summer-dress",
    confidence=0.85,
    source="rule"
)

# Create request
request = TagGenerationRequest(
    product_data=product,
    max_tags=10
)
```

## Testing

### Test Suite (`test_hirusha_components.py`)
Comprehensive test suite that validates all implemented components:

- **TagGeneratorAgent Tests**: Tests tag generation with various product data
- **Security Tests**: Tests password hashing, JWT tokens, input sanitization, CSRF protection
- **Schema Tests**: Tests all Pydantic models and validation

**Running Tests**:
```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python test_hirusha_components.py
```

**Test Results**:
```
🚀 Starting Hirusha's Component Tests
==================================================
🧪 Testing TagGeneratorAgent...
✅ Tag generation successful!
   Generated 4 tags
   Timestamp: 2025-09-23T10:29:38.949420
   Sample tags:
     1. dress (confidence: 0.90)
     2. red-color (confidence: 0.80)
     3. summer-season (confidence: 0.70)

🔒 Testing Security utilities...
✅ Password hashing: PASS
✅ Password validation: PASS
   Score: 5/5
✅ JWT tokens: PASS
✅ Input sanitization: PASS
✅ CSRF tokens: PASS
✅ Filename safety: PASS

📋 Testing API Schemas...
✅ ProductData schema: PASS
✅ Tag schema: PASS
✅ TagGenerationRequest schema: PASS
✅ TagGenerationResponse schema: PASS
✅ ErrorResponse schema: PASS
✅ HealthCheck schema: PASS

==================================================
📊 Test Summary:
✅ Passed: 3/3
❌ Failed: 0/3
🎉 All tests passed! Your components are working correctly.
```

## Dependencies

### Required Packages
- `fastapi`: Web framework
- `pydantic`: Data validation
- `passlib`: Password hashing
- `python-jose`: JWT token handling
- `bcrypt`: Password hashing backend
- `python-dotenv`: Environment variable management

### Installation
```bash
pip install -r backend/requirements.txt
```

## Integration Notes

### TagGeneratorAgent Integration
- Can work independently or with LLM/NLP processors
- Graceful degradation when external services are unavailable
- Returns structured data compatible with API schemas

### Security Integration
- JWT tokens can be used for API authentication
- Password validation can be integrated into user registration
- Input sanitization should be applied to all user inputs

### Schema Integration
- All schemas are compatible with FastAPI
- Automatic API documentation generation
- Request/response validation

## File Structure
```
backend/
├── agents/
│   └── tag_generator_agent.py      # Tag generation logic
├── core/
│   └── security.py                 # Security utilities
├── api/
│   └── schemas.py                  # API data models
└── requirements.txt                # Dependencies

test_hirusha_components.py          # Test suite
README_HIRUSHA.md                   # This documentation
```

## Development Status
✅ **COMPLETED**: All assigned components implemented and tested
- TagGeneratorAgent: Fully functional with multiple tag generation approaches
- Security Module: Comprehensive security utilities implemented
- API Schemas: Complete Pydantic models with validation
- Test Suite: All components tested and working correctly

## Next Steps
1. **Integration**: Integrate with other team members' components
2. **API Endpoints**: Create FastAPI endpoints using the schemas
3. **Database Integration**: Add database persistence for tags and security data
4. **Performance Optimization**: Optimize tag generation for large datasets
5. **Monitoring**: Add logging and monitoring for production use

## Contact
**Hirusha** - IT23426580
- Components: TagGeneratorAgent, Security Module, API Schemas
- Status: ✅ Complete and Tested
