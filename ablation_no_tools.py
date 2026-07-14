import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
import json
import base64
import logging
import argparse
import asyncio
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global variables
logger = None
temp_dir_path = None

# Configuration
# Set to a list of question IDs to re-run only specific questions; None runs all
RETRY_IDS = None
PRELOAD_RESULTS = None  # path to a pre-answered results_summary.json to resume from
# Parallel batching: set BATCH_TOTAL > 1 and launch BATCH_TOTAL copies with BATCH_INDEX 0..N-1
BATCH_TOTAL = 1
BATCH_INDEX = 0

MODEL_CONFIGS = {
    'eve': {
        'model_name': 'EVE-Instruct',
        'endpoint_env': 'EVE_ENDPOINT',
        'port_replace': None,
    },
    'mistral-small': {
        'model_name': 'Mistral-Small-3.2-24B-Instruct-2506',
        'endpoint_env': 'MISTRAL_ENDPOINT',
        'port_replace': None,
    },
}

sys_prompt = '''
You are a geoscientist answering multiple-choice questions about Earth observation data analysis. You do NOT have access to any tools, files, or external data in this task - you must answer using only your own internal knowledge and reasoning about the question text below.
ATTENTION:
1. You must still provide the choice you think is most appropriate, even without access to the underlying data.
2. Your final answer format must be:
<Answer>Your choice<Answer>
'''

username = os.getenv("EVE_USERNAME")
password = os.getenv("EVE_PASSWORD")
token = base64.b64encode(f"{username}:{password}".encode()).decode()
headers = {"Authorization": f"Basic {token}"}


def init_global_params(model_name):
    """Initialize global parameters and logging"""
    global temp_dir_path, logger

    if temp_dir_path is None:
        batch_suffix = f'_b{BATCH_INDEX}of{BATCH_TOTAL}' if BATCH_TOTAL > 1 else ''
        temp_dir_path = Path('./evaluate_langchain/{}_IF_notools_{}{}'.format(
            model_name,
            datetime.now().strftime('%y-%m-%d_%H-%M'),
            batch_suffix
        )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "question_index": record.args[0] if record.args else "unknown",
                "timestamp": self.formatTime(record, self.datefmt),
                "conversations": record.args[1] if len(record.args) > 1 else [],
                "final_answer": record.args[2] if len(record.args) > 2 else None
            }
            return json.dumps(log_record, ensure_ascii=False, indent=4)

    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        temp_dir_path / "{}_IF_notools_langchain.log".format(model_name)
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return temp_dir_path, logger


def init_chat_logger(model_name):
    """Initialize chat logger for .chat file like AgentScope"""
    global temp_dir_path
    chat_log_path = temp_dir_path / "{}_IF_notools_langchain.chat".format(model_name)
    return chat_log_path


def save_chat_message(chat_log_path, message_data):
    import uuid
    chat_record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage",
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'langchain_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": message_data.get('content', []),
        "metadata": message_data.get('metadata', None),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')


def load_llm(model_key):
    """Build the ChatOpenAI client for the selected model, no tools attached"""
    cfg = MODEL_CONFIGS[model_key]
    base_url = os.getenv(cfg['endpoint_env'])
    if cfg['port_replace']:
        base_url = base_url.replace(*cfg['port_replace'])

    llm_kwargs = {
        'model': cfg['model_name'],
        'api_key': "EMPTY",
        'base_url': base_url,
        'temperature': 0,
        'max_tokens': 2048,
        'request_timeout': 300,
        'default_headers': headers,
    }
    return ChatOpenAI(**llm_kwargs)


def load_questions(test_json_path: str = 'benchmark/question.json'):
    """Load evaluation questions (IF phrasing only)"""
    with open(test_json_path, 'r') as f:
        test_json = json.load(f)

    out = []
    for _, (question_idx, question_info) in enumerate(test_json.items()):
        AP_INDEX = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
        IF_INDEX = 1 - AP_INDEX
        data = question_info['evaluation'][IF_INDEX].get('data', None)
        data = question_info['evaluation'][AP_INDEX].get('data', None) if data is None else data

        if data is None:
            continue
        out.append({
            "question_id": question_idx,
            "instruct": question_info['evaluation'][IF_INDEX]['question'],
            "data": data,
            "choices": question_info.get('choices', None)
        })

    return out


async def handle_question(llm, question, chat_log_path):
    """Handle a single question with a direct, single-turn, tool-free LLM call"""
    try:
        query = question['instruct'] + question['data']

        if question['choices']:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice)
                for i, choice in enumerate(question['choices'])
            ])

        full_query = f"{sys_prompt}\n\nQuestion: {query}"

        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")

        user_message = {
            "name": "user",
            "role": "user",
            "content": full_query,
            "metadata": {"question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, user_message)

        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=full_query)]),
            timeout=300,  # single-turn call, no tool loop, generous cap is enough
        )

        final_answer = response.content

        conversation_log = [
            {"role": "user", "content": full_query},
            {"role": "assistant", "content": [{"type": "text", "content": final_answer}]},
        ]

        assistant_message = {
            "name": question['question_id'],
            "role": "assistant",
            "content": [{"type": "text", "text": final_answer}],
            "metadata": None
        }
        save_chat_message(chat_log_path, assistant_message)

        logger.info("Chat Content", question['question_id'], conversation_log, final_answer)

        print(f"Final Answer: {final_answer}")
        return final_answer

    except Exception as e:
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)

        error_message = {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, error_message)

        logger.info(question['question_id'], [], error_msg)
        return f"Error: {e}"


async def main(model_key):
    print(f"Initializing no-tools ablation for model: {model_key}")

    init_global_params(model_key)
    chat_log_path = init_chat_logger(model_key)
    print(f"Chat log will be saved to: {chat_log_path}")

    llm = load_llm(model_key)

    print("Warming up connection...")
    try:
        await asyncio.wait_for(llm.ainvoke("Hi"), timeout=120)
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up failed (continuing anyway): {e}")

    questions = load_questions()
    if RETRY_IDS is not None:
        retry_set = set(RETRY_IDS)
        questions = [q for q in questions if q['question_id'] in retry_set]
    if BATCH_TOTAL > 1:
        questions = [q for i, q in enumerate(questions) if i % BATCH_TOTAL == BATCH_INDEX]
    print(f"Loaded {len(questions)} questions for evaluation"
          + (f" (batch {BATCH_INDEX+1}/{BATCH_TOTAL})" if BATCH_TOTAL > 1 else ""))

    results = []
    if PRELOAD_RESULTS and Path(PRELOAD_RESULTS).exists():
        with open(PRELOAD_RESULTS) as _f:
            results = json.load(_f)
        print(f"Preloaded {len(results)} existing answers from {PRELOAD_RESULTS}")

    for question in tqdm(questions, desc="Processing questions"):
        answer = await handle_question(llm, question, chat_log_path)
        results.append({
            "question_id": question['question_id'],
            "answer": answer
        })

    results_path = temp_dir_path / "results_summary.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation completed! Results saved to {results_path}")
    print(f"Detailed logs available at: {temp_dir_path}")
    print(f"Chat history saved to: {chat_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-tools ablation study for EVE / Mistral-Small")
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True,
                         help="Which vLLM-served model to run without tool access")
    args = parser.parse_args()
    asyncio.run(main(args.model))
