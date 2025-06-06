from flask import Flask, request, jsonify
"""
A Flask API that provides access to AWS Bedrock's Claude models through a chat completions endpoint.
The API mimics OpenAI's chat completions interface but routes requests to AWS Bedrock Claude models.
Supports Claude 3.5 Sonnet, 3.7 Sonnet, 4 Sonnet and 4 Opus models.
Example curl usage:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "claude-3.5-sonnet",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Speak like a poet."},
                {"role": "user", "content": "What is your model name and cutoff date?"},
                {"role": "assistant", "content": "I am Claude 3.5 Sonnet, trained on data up to September 2023."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 256
        }'
Request Parameters:
    model (str): The model ID to use. Defaults to 'claude-3.5-sonnet'.
                 Options: claude-3.5-sonnet, claude-3.7-sonnet, claude-4-sonnet, claude-4-opus
    messages (list): List of message objects with 'role' and 'content' fields
    max_tokens (int): Maximum number of tokens to generate. Defaults to 256
Returns:
    JSON response matching OpenAI's chat completion format:
    {
        "id": str,              # Unique identifier for the completion
        "object": str,          # Object type "chat.completion"
        "created": int,         # Timestamp of when the completion was created
        "model": str,           # Model used for completion
            "index": int,       # Always 0 for single completion
                "role": str,    # "assistant"
                "content": str  # Generated response
            "finish_reason": str # Reason for finishing
Raises:
    400: If an invalid model is specified
    500: For other errors during processing
"""
import boto3
import json
import os

app = Flask(__name__)

def get_bedrock_client():
    sts = boto3.client("sts")
    response = sts.assume_role(
        RoleArn="arn:aws:iam::396608793503:role/BedrockReadOnly",
        RoleSessionName="aira",
    )
    creds = response["Credentials"]
    s = boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name="us-west-2",
    )
    return s.client(service_name="bedrock-runtime")

MODEL_MAP = {
    "claude-3.5-sonnet": "arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3.7-sonnet": "arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-4-sonnet": "arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-4-opus": "arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.anthropic.claude-opus-4-20250514-v1:0"
}

@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        model = data.get('model', 'claude-3.5-sonnet')
        for key in MODEL_MAP.keys():
            if key in model:
                model = key
                break
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 200000)
        print(f"Max tokens: {max_tokens}")

        if model not in MODEL_MAP:
            return jsonify({"error": "Invalid model"}), 400

        bedrock = get_bedrock_client()
        new_messages = []
        system_message = None
        for message in messages:
            if 'role' in message and message['role'] == 'system':
                system_message = message['content']
                continue
            new_messages.append(message)
        messages = new_messages
        
        body = {
            "max_tokens": max_tokens,
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
        }
        if system_message:
            body["system"] = system_message
        body = json.dumps(body)

        response = bedrock.invoke_model(
            body=body,
            modelId=MODEL_MAP[model]
        )
        response_body = json.loads(response.get("body").read())
        # __import__("ipdb").set_trace()  # For debugging purposes, remove in production
        created_timestamp = response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date", "0")
        if not created_timestamp:
            created_timestamp = 0
        else:
            from datetime import datetime
            created_timestamp = int(datetime.strptime(created_timestamp, '%a, %d %b %Y %H:%M:%S GMT').timestamp())
            created_timestamp = created_timestamp

        print(f"Response body: \n{response_body.get('content', '')}")

        return jsonify({
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": created_timestamp,
            "model": "openai/" + model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_body.get("content", "")[0].get("text", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response_body.get("usage", {}).get("output_tokens", 0),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000)
