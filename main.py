import os
import re
from dotenv import load_dotenv
from guardrails import Guard
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Custom Validation Functions ---

def check_toxic_language(text, threshold=0.8):
    toxic_words = ['stupid', 'idiot', 'hate', 'kill', 'die', 'toxic']
    text_lower = text.lower()
    toxic_count = sum(1 for word in toxic_words if word in text_lower)
    return toxic_count == 0

def detect_pii(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'

    pii_found = []
    if re.search(email_pattern, text):
        pii_found.append("EMAIL")
    if re.search(phone_pattern, text):
        pii_found.append("PHONE")
    if re.search(ssn_pattern, text):
        pii_found.append("SSN")

    return pii_found

def check_sql_injection(text):
    sql_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'SELECT', 'UNION', 'ALTER', 'CREATE']
    text_upper = text.upper()
    for keyword in sql_keywords:
        if keyword in text_upper:
            return False
    return True

def check_prompt_injection(text):
    injection_patterns = [
        'ignore previous instructions',
        'system prompt',
        'you are now',
        'forget everything',
        'new instructions'
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if pattern in text_lower:
            return False
    return True

def check_competitor_mentions(text, competitors=None):
    if competitors is None:
        competitors = ['competitor1', 'competitor2']
    text_lower = text.lower()
    for competitor in competitors:
        if competitor.lower() in text_lower:
            return False
    return True

def filter_pii(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    
    text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)
    text = re.sub(phone_pattern, '[PHONE_REDACTED]', text)
    text = re.sub(ssn_pattern, '[SSN_REDACTED]', text)
    
    return text

def check_token_limit(text, max_tokens=150):
    estimated_tokens = len(text) / 4
    return estimated_tokens <= max_tokens

def check_coherency(text):
    words = text.split()
    if len(words) < 3:
        return False
    if len(set(words)) < len(words) * 0.3:
        return False
    return True

def run_moderation_check(text):
    try:
        moderation_resp = client.moderations.create(input=text)
        results = moderation_resp.results[0]
        if results.flagged:
            flagged_categories = [
                cat for cat, flagged in results.categories.items() if flagged
            ]
            raise Exception(f"Content flagged by moderation API for: {', '.join(flagged_categories)}")
    except Exception as e:
        print(f"Moderation API error: {e}")
        # Proceed without blocking if moderation fails

def validate_input(message):
    errors = []
    warnings = []

    if not check_toxic_language(message):
        errors.append("Toxic language detected")

    if not check_sql_injection(message):
        errors.append("SQL injection detected")

    if not check_prompt_injection(message):
        errors.append("Prompt injection detected")

    if not check_competitor_mentions(message):
        warnings.append("Competitor mention detected")

    pii_found = detect_pii(message)
    if pii_found:
        warnings.append(f"PII detected: {', '.join(pii_found)}")

    return errors, warnings

def validate_output(response):
    errors = []
    warnings = []

    if not check_toxic_language(response):
        warnings.append("Toxic language in output")

    if not check_sql_injection(response):
        warnings.append("SQL injection in output")

    pii_found = detect_pii(response)
    if pii_found:
        warnings.append(f"PII in output: {', '.join(pii_found)}")

    return errors, warnings

# --- Main Guardrails Chat Function ---

def chat_with_guardrails(message, max_tokens=150):
    warnings = []

    try:
        print("🔍 Checking input guardrails...")

        if not check_token_limit(message, max_tokens):
            raise Exception("Input exceeds token limit")

        if not check_coherency(message):
            warnings.append("Input coherency warning")

        input_errors, input_warnings = validate_input(message)
        if input_errors:
            raise Exception(f"Input validation failed: {', '.join(input_errors)}")
        warnings.extend(input_warnings)

        run_moderation_check(message)

        filtered_message = filter_pii(message)
        print("✅ Input guardrails passed")

        print("🤖 Calling OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": filtered_message}],
            max_tokens=max_tokens
        )
        ai_response = response.choices[0].message.content

        print("🔍 Checking output guardrails...")

        if not check_coherency(ai_response):
            warnings.append("Output coherency warning")

        run_moderation_check(ai_response)

        output_errors, output_warnings = validate_output(ai_response)
        if output_errors:
            raise Exception(f"Output validation failed: {', '.join(output_errors)}")
        warnings.extend(output_warnings)

        filtered_output = filter_pii(ai_response)
        print("✅ Output guardrails passed")

        return {
            "response": filtered_output,
            "guardrails_passed": True,
            "warnings": warnings
        }

    except Exception as e:
        print(f"❌ Guardrails violation: {str(e)}")
        return {
            "response": None,
            "guardrails_passed": False,
            "error": str(e)
        }

# --- Example Usage ---

if __name__ == "__main__":
    test_messages = [
        "Hello, how are you?",
        "DROP TABLE users;",
        "You are stupid and toxic",
        "My email is john@example.com and phone is 123-456-7890",
        "I want to hurt myself",
    ]

    for msg in test_messages:
        print("\n" + "=" * 50)
        print(f"Testing: {msg}")
        print("=" * 50)
        result = chat_with_guardrails(msg)

        if result["guardrails_passed"]:
            print(f"✅ Response: {result['response']}")
            if result["warnings"]:
                print(f"⚠ Warnings: {result['warnings']}")
        else:
            print(f"🚫 Blocked: {result['error']}")
