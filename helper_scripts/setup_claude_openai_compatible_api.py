# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A Flask API that provides access to AWS Bedrock's Claude models through a chat completions endpoint.
The API mimics OpenAI's chat completions interface but routes requests to AWS Bedrock Claude models.
Supports Claude 3.5 Sonnet, 3.7 Sonnet, 4 Sonnet and 4 Opus models.
To spin up the proxy server, run:
    gunicorn -w 16 -b 0.0.0.0:8000 setup_claude_openai_compatible_api:app --timeout 1800
The --timeout argument is important because Claude takes a long time to respond to long prompts. After 
spinning the server, point the model_url to http://localhost:8000/v1 
This API supports both streaming and non-streaming responses; streaming generation prevents timeouts as 
streaming will return one token as soon as it is generated.
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
from flask import Flask, request, jsonify, Response
import boto3
from botocore.config import Config
import json
import os
import time

app = Flask(__name__)

def get_bedrock_client():
    config = Config(read_timeout=60*60)
    sts = boto3.client("sts", config=config)
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

import tiktoken
def count_tokens(text, model="gpt-4"):
    """Count tokens using OpenAI's tiktoken library"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback to character-based counting if tiktoken fails
        return len(text) // 6

def truncate_to_token_limit(body, max_tokens=121070):
    """
    Truncate the conversation to keep only the most recent tokens within the limit.
    Assumes body contains messages in OpenAI format.
    """
    if not isinstance(body, dict) or 'messages' not in body:
        return body
    
    messages = body['messages']
    if not messages:
        return body
    
    # Calculate tokens for each message
    message_tokens = []
    total_tokens = 0
    
    # Process messages in reverse order (most recent first)
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        content = message.get('content', '')
        if isinstance(content, list):
            # Handle multi-part content (text + images)
            text_content = ''
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_content += part.get('text', '')
        else:
            text_content = str(content)
        
        tokens = count_tokens(text_content)
        message_tokens.append((i, tokens, message))
        total_tokens += tokens
    
    # If within limit, return original body
    if total_tokens <= max_tokens:
        print(f"Total tokens {total_tokens} within limit {max_tokens}. No truncation needed.")
        return body
    
    # Truncate messages, keeping the most recent ones
    kept_messages = []
    current_tokens = 0
    
    # Always keep the system message if it exists and is first
    system_message = None
    if 'system' in body:
        # system_message = messages[0]
        system_message = body['system']
        system_tokens = count_tokens(str(system_message))
        current_tokens += system_tokens
    
    # Add messages from most recent, staying within token limit
    for i, tokens, message in message_tokens:
        if current_tokens + tokens <= max_tokens:
            kept_messages.append((i, message))
            current_tokens += tokens
        else:
            break
    
    # Sort kept messages by original index to maintain conversation order
    kept_messages.sort(key=lambda x: x[0])
    
    # Reconstruct the body with truncated messages
    new_messages = []
    if system_message:
        # new_messages.append(system_message)
        body['system'] = system_message
    
    new_messages.extend([msg for _, msg in kept_messages])
    
    truncated_body = body.copy()
    truncated_body['messages'] = new_messages
    
    print(f"Truncated conversation: {len(messages)} -> {len(new_messages)} messages")
    print(f"Token count: {total_tokens} -> {current_tokens} tokens")
    
    return truncated_body

def generate_streaming_response(bedrock, body, model_id, model_name):
    """Generate streaming response from Bedrock"""
    try:
        response = bedrock.invoke_model_with_response_stream(
            body=body,
            modelId=model_id
        )
        
        completion_id = "chatcmpl-" + os.urandom(12).hex()
        created_timestamp = int(time.time())
        
        # Send initial chunk
        initial_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": "openai/" + model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        # Process streaming response
        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    
                    if chunk_obj['type'] == 'content_block_delta':
                        delta_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": "openai/" + model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk_obj['delta']['text']},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"
                    
                    elif chunk_obj['type'] == 'message_stop':
                        # Send final chunk
                        final_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": "openai/" + model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        break
        
        # Send [DONE] marker
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

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
        max_tokens = data.get('max_tokens', 8192)
        stream = data.get('stream', False)
        
        print(f"Max tokens: {max_tokens}, Stream: {stream}")

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
        body = truncate_to_token_limit(body)
        body_json = json.dumps(body)

        if stream:
            # Return streaming response
            return Response(
                generate_streaming_response(bedrock, body_json, MODEL_MAP[model], model),
                content_type='text/plain; charset=utf-8',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        else:
            # Non-streaming response (original behavior)
            response = bedrock.invoke_model(
                body=body_json,
                modelId=MODEL_MAP[model]
            )
            response_body = json.loads(response.get("body").read())
            
            created_timestamp = response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date", "0")
            if not created_timestamp:
                created_timestamp = 0
            else:
                from datetime import datetime
                created_timestamp = int(datetime.strptime(created_timestamp, '%a, %d %b %Y %H:%M:%S GMT').timestamp())

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
        if stream:
            def error_stream():
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            return Response(error_stream(), content_type='text/plain; charset=utf-8')
        else:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000)
